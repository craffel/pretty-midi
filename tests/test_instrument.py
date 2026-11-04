import numpy as np
import pytest

import pretty_midi


@pytest.mark.parametrize('notes,use_duration,use_velocity', [
    ([], False, False),
    ([pretty_midi.Note(100, 60, 1., 1.)], True, False),
    ([pretty_midi.Note(0, 60, 0., 1.)], False, True),
    ([pretty_midi.Note(0, 60, 1., 1.)], True, True),
])
def test_normalized_histogram_with_zero_weight(
        notes, use_duration, use_velocity):
    instrument = pretty_midi.Instrument(0)
    instrument.notes = notes
    with np.errstate(divide='raise', invalid='raise'):
        histogram = instrument.get_pitch_class_histogram(
            use_duration, use_velocity, normalize=True)
    np.testing.assert_array_equal(histogram, np.zeros(12))


@pytest.mark.parametrize('use_duration,use_velocity,weights', [
    (False, False, [2., 1.]),
    (True, False, [3., 1.]),
    (False, True, [150., 100.]),
    (True, True, [200., 100.]),
])
def test_histogram_normalization_preserves_weights(
        use_duration, use_velocity, weights):
    instrument = pretty_midi.Instrument(0)
    instrument.notes = [pretty_midi.Note(100, 60, 0., 1.),
                        pretty_midi.Note(50, 72, 2., 4.),
                        pretty_midi.Note(100, 64, 4., 5.)]
    expected = np.zeros(12)
    expected[[0, 4]] = weights
    np.testing.assert_array_equal(instrument.get_pitch_class_histogram(
        use_duration, use_velocity, normalize=False), expected)
    np.testing.assert_allclose(instrument.get_pitch_class_histogram(
        use_duration, use_velocity, normalize=True), expected / expected.sum())


def test_normalized_transition_matrix_without_transitions():
    instrument = pretty_midi.Instrument(0)
    instrument.notes = [pretty_midi.Note(100, 60, 0., 1.),
                        pretty_midi.Note(100, 64, 2., 3.)]
    with np.errstate(divide='raise', invalid='raise'):
        matrix = instrument.get_pitch_class_transition_matrix(normalize=True)
    np.testing.assert_array_equal(matrix, np.zeros((12, 12)))


def test_transition_matrix_normalization_preserves_counts():
    instrument = pretty_midi.Instrument(0)
    instrument.notes = [pretty_midi.Note(100, 60, 0., 1.),
                        pretty_midi.Note(100, 64, 1., 2.),
                        pretty_midi.Note(100, 72, 2., 3.),
                        pretty_midi.Note(100, 64, 3., 4.)]
    expected = np.zeros((12, 12))
    expected[0, 4] = 2.
    expected[4, 0] = 1.
    np.testing.assert_array_equal(instrument.get_pitch_class_transition_matrix(
        normalize=False), expected)
    np.testing.assert_allclose(instrument.get_pitch_class_transition_matrix(
        normalize=True), expected / expected.sum())


def test_normalized_transition_matrix_with_silent_instrument():
    silent = pretty_midi.Instrument(0)
    silent.notes = [pretty_midi.Note(100, 60, 0., 1.),
                    pretty_midi.Note(100, 64, 2., 3.)]
    active = pretty_midi.Instrument(0)
    active.notes = [pretty_midi.Note(100, 60, 0., 1.),
                    pretty_midi.Note(100, 64, 1., 2.)]
    midi = pretty_midi.PrettyMIDI()
    midi.instruments = [silent, active]
    with np.errstate(divide='raise', invalid='raise'):
        matrix = midi.get_pitch_class_transition_matrix(normalize=True)
    expected = np.zeros((12, 12))
    expected[0, 4] = 1.
    np.testing.assert_array_equal(matrix, expected)


def test_remove_invalid_notes_keeps_order_and_list():
    instrument = pretty_midi.Instrument(0)
    notes = [pretty_midi.Note(100, 60 + i, i, i + 1) for i in range(6)]
    notes[1].end = notes[1].start
    notes[4].end = notes[4].start - .5
    instrument.notes = notes + [notes[1]]
    original_list = instrument.notes
    instrument.remove_invalid_notes()
    assert instrument.notes is original_list
    assert instrument.notes == [notes[0], notes[2], notes[3], notes[5]]


@pytest.mark.parametrize('bend_times', [[.5], [.25, .5, .75], [0., .5, 1.]])
def test_synthesize_pitch_bend_keeps_phase_continuous(bend_times):
    fs = 8000
    instrument = pretty_midi.Instrument(0)
    instrument.notes = [pretty_midi.Note(1, 69, 0., 1.)]
    instrument.pitch_bends = [
        pretty_midi.PitchBend(4096 * (i % 2 * 2 - 1), t)
        for i, t in enumerate(bend_times)]
    waveform = instrument.synthesize(fs=fs)[:fs]
    # Without the per-bend offset the phase would jump at each bend
    max_step = 2 * np.pi * pretty_midi.note_number_to_hz(70) / fs
    assert np.abs(np.diff(waveform)).max() < 1.05 * max_step


# Verbatim copies of the implementations before the speed-ups, used as
# references below. Only `self` and the helper names are changed.
def _legacy_remove_invalid_notes(instrument):
    # Crete a list of all invalid notes
    notes_to_delete = []
    for note in instrument.notes:
        if note.end <= note.start:
            notes_to_delete.append(note)
    # Remove the notes found
    for note in notes_to_delete:
        instrument.notes.remove(note)


def _legacy_synthesize(instrument, fs=44100, wave=np.sin):
    # Pre-allocate output waveform
    synthesized = np.zeros(int(fs*(instrument.get_end_time() + 1)))

    # If we're a percussion channel, just return the zeros
    if instrument.is_drum:
        return synthesized
    # If the above if statement failed, we need to revert back to default
    if not hasattr(wave, '__call__'):
        raise ValueError('wave should be a callable Python function')
    # This is a simple way to make the end of the notes fade-out without
    # clicks
    fade_out = np.linspace(1, 0, int(.1*fs))
    # Create a frequency multiplier array for pitch bend
    bend_multiplier = np.ones(synthesized.shape)
    # Need to sort the pitch bend list for the loop below to work
    ordered_bends = sorted(instrument.pitch_bends, key=lambda bend: bend.time)
    # Add in a bend of 0 at the end of time
    end_bend = pretty_midi.PitchBend(0, instrument.get_end_time())
    for start_bend, end_bend in zip(ordered_bends,
                                    ordered_bends[1:] + [end_bend]):
        # Bend start and end time in samples
        start = int(start_bend.time*fs)
        end = int(end_bend.time*fs)
        # The multiplier will be (twelfth root of 2)^(bend semitones)
        bend_semitones = pretty_midi.pitch_bend_to_semitones(start_bend.pitch)
        bend_amount = (2**(1/12.))**bend_semitones
        # Sample indices effected by the bend
        bend_multiplier[start:end] = bend_amount
    # Add in waveform for each note
    for note in instrument.notes:
        # Indices in samples of this note
        start = int(fs*note.start)
        end = int(fs*note.end)
        # Get frequency of note from MIDI note number
        frequency = pretty_midi.note_number_to_hz(note.pitch)
        # When a pitch bend gets applied, there will be a sample
        # discontinuity. So, we also need an array of offsets which get
        # applied to compensate.
        offsets = np.zeros(end - start)
        for bend in ordered_bends:
            bend_sample = int(bend.time*fs)
            # Does this pitch bend fall within this note?
            if bend_sample > start and bend_sample < end:
                # Compute the average bend so far
                bend_so_far = bend_multiplier[start:bend_sample].mean()
                bend_amount = bend_multiplier[bend_sample]
                # Compute the offset correction
                offset = (bend_so_far - bend_amount)*(bend_sample - start)
                # Store this offset for samples effected
                offsets[bend_sample - start:] = offset
        # Compute the angular frequencies, bent, over this interval
        frequencies = 2*np.pi*frequency*(bend_multiplier[start:end])/fs
        # Synthesize using wave function at this frequency
        note_waveform = wave(frequencies*np.arange(end - start) +
                             2*np.pi*frequency*offsets/fs)
        # Apply an exponential envelope
        envelope = np.exp(-np.arange(end - start)/(1.0*fs))
        # Make the end of the envelope be a fadeout
        if envelope.shape[0] > fade_out.shape[0]:
            envelope[-fade_out.shape[0]:] *= fade_out
        else:
            envelope *= np.linspace(1, 0, envelope.shape[0])
        # Multiply by velocity (don't think it's linearly scaled but
        # whatever)
        envelope *= note.velocity
        # Add in envelope'd waveform to the synthesized signal
        synthesized[start:end] += envelope*note_waveform

    return synthesized


def _random_instrument(rng, n_notes, n_bends):
    """Notes with zero and negative durations and one listed twice, and
    bends that may fall exactly on note boundaries."""
    grid = .01
    instrument = pretty_midi.Instrument(0)
    for _ in range(n_notes):
        start = rng.integers(0, 200) * grid
        end = start + rng.integers(-3, 30) * grid
        note = pretty_midi.Note(
            int(rng.integers(1, 128)), int(rng.integers(30, 90)),
            start, start + 1.)
        # The constructor rejects invalid durations, so set the end after
        note.end = max(end, 0.)
        instrument.notes.append(note)
    if instrument.notes:
        instrument.notes.append(
            instrument.notes[rng.integers(len(instrument.notes))])
    note_times = [t for note in instrument.notes
                  for t in (note.start, note.end)]
    for _ in range(n_bends):
        if note_times and rng.random() < .3:
            time = note_times[rng.integers(len(note_times))]
        else:
            time = rng.integers(0, 230) * grid
        instrument.pitch_bends.append(
            pretty_midi.PitchBend(int(rng.integers(-8192, 8192)), time))
    return instrument


@pytest.mark.parametrize('seed', range(200))
def test_remove_invalid_notes_matches_legacy(seed):
    rng = np.random.default_rng(seed)
    instrument = _random_instrument(rng, int(rng.integers(0, 40)), 0)
    legacy = pretty_midi.Instrument(0)
    legacy.notes = list(instrument.notes)
    _legacy_remove_invalid_notes(legacy)
    expected = legacy.notes
    instrument.remove_invalid_notes()
    assert len(instrument.notes) == len(expected)
    assert all(a is b for a, b in zip(instrument.notes, expected))


@pytest.mark.parametrize('seed', range(100))
@pytest.mark.parametrize('fs', [800, 8000, 22050])
def test_synthesize_matches_legacy(seed, fs):
    rng = np.random.default_rng(seed)
    instrument = _random_instrument(
        rng, int(rng.integers(1, 15)), int(rng.integers(0, 25)))
    instrument.remove_invalid_notes()
    expected = _legacy_synthesize(instrument, fs=fs)
    actual = instrument.synthesize(fs=fs)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
