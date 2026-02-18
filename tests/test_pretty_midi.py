import pretty_midi
import pytest
import numpy as np
import mido
from tempfile import NamedTemporaryFile


def test_pm_object_initialization():
    def make_mido_track(notes_str):
        track = mido.MidiTrack()
        for line in notes_str.split('\n'):
            line = line.strip()
            if line:
                track.append(mido.Message.from_str(line))
        mido_obj = mido.MidiFile()
        mido_obj.tracks.append(track)
        return mido_obj

    example_track_1 = """
    note_on channel=0 note=72 velocity=88 time=0
    note_on channel=0 note=72 velocity=0 time=48
    note_on channel=0 note=72 velocity=88 time=0
    note_on channel=0 note=74 velocity=88 time=48
    note_on channel=0 note=72 velocity=0 time=0
    note_on channel=0 note=72 velocity=88 time=48
    note_on channel=0 note=74 velocity=0 time=0
    note_on channel=0 note=72 velocity=0 time=48
    """

    example_track_2 = """
    note_on channel=0 note=72 velocity=88 time=0
    note_on channel=0 note=72 velocity=0 time=48
    note_on channel=0 note=72 velocity=88 time=0
    note_on channel=0 note=74 velocity=88 time=48
    note_on channel=0 note=72 velocity=0 time=0
    note_on channel=0 note=72 velocity=88 time=48
    note_on channel=0 note=74 velocity=0 time=0
    note_on channel=0 note=72 velocity=0 time=48
    note_on channel=0 note=75 velocity=88 time=0
    note_on channel=0 note=75 velocity=0 time=48
    """

    # Test-1: Passing pre-loaded mido.MidiFile object
    example_mido_obj_1 = make_mido_track(example_track_1)
    pm_song = pretty_midi.PrettyMIDI(mido_object=example_mido_obj_1)
    assert len(pm_song.instruments[0].notes) == 4

    # Test-2: Testing value error is raised when non mido.MidiFile object is passed as a mido_object argument
    try:
        pm_song = pretty_midi.PrettyMIDI(mido_object=mido.MidiTrack())
        raise Exception("Expected ValueError when non mido.MidiFile object is passed as a mido_object argument.")
    except ValueError as val_error:
        assert val_error.args[0] == "Expected mido_object to be of type mido.MidiFile."

    with NamedTemporaryFile() as file:

        # Test-3: Passing file path while mido_object argument defaults to None.
        # This test will ensure <=v0.2.10 compatibility for passing other arguments without keywords
        example_mido_obj_2 = make_mido_track(example_track_2)
        example_mido_obj_2.save(file=file)
        file.seek(0)
        pm_song = pretty_midi.PrettyMIDI(file)
        assert len(pm_song.instruments[0].notes) == 5

        # Test-4: Testing value error is raised when both midi_file and mido_object arguments are provided
        try:
            pm_song = pretty_midi.PrettyMIDI(midi_file=file, mido_object=example_mido_obj_1)
            raise Exception("Expected ValueError when both midi_file and mido_object arguments are provided.")
        except ValueError as val_error:
            assert val_error.args[0] == ("Either the midi_file or the mido_object argument must be provided, "
                                         "but not both.")


def test_get_beats():
    pm = pretty_midi.PrettyMIDI()
    # Add a note to force get_end_time() to be non-zero
    i = pretty_midi.Instrument(0)
    i.notes.append(pretty_midi.Note(100, 100, 0.3, 10.4))
    pm.instruments.append(i)
    # pretty_midi assumes 120 bpm unless otherwise specified
    assert np.allclose(pm.get_beats(),
                       np.arange(0, pm.get_end_time(), 60./120.))
    # Testing starting from a different beat time
    assert np.allclose(pm.get_beats(.2),
                       np.arange(0, pm.get_end_time(), 60./120.) + .2)
    # Testing a tempo change
    change_bpm = 93.
    change_time = 4.4
    pm._tick_scales.append(
        (pm.time_to_tick(change_time), 60./(change_bpm*pm.resolution)))
    pm._update_tick_to_time(pm.time_to_tick(pm.get_end_time()))
    # Track at 120 bpm up to the tempo change time
    expected_beats = np.arange(0, change_time, 60./120.)
    # BPM switches (4.5 - 4.4)/(60./120.) of the way through
    expected_beats = np.append(
        expected_beats,
        change_time + (4.5 - change_time)/(60./120.)*60./change_bpm)
    # From there, use the new bpm
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 60./change_bpm,
                                         pm.get_end_time(), 60./change_bpm))
    assert np.allclose(pm.get_beats(), expected_beats)
    # When requesting a start_time after the tempo change, make sure we just
    # track as normal
    assert np.allclose(
        pm.get_beats(change_time + .1),
        np.arange(change_time + .1, pm.get_end_time(), 60./change_bpm))
    # Add a time signature change, which forces beat tracking to restart
    pm.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 2.1))
    # Track at 120 bpm up to time signature change
    expected_beats = np.arange(0, 2.1, 60./120.)
    # Now track, restarting from time signature change time
    expected_beats = np.append(expected_beats,
                               np.arange(2.1, change_time, 60./120.))
    # BPM switches (4.6 - 4.4)/(60./120.) of the way through
    expected_beats = np.append(
        expected_beats,
        change_time + (4.6 - change_time)/(60./120.)*60./change_bpm)
    # From there, use the new bpm
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 60./change_bpm,
                                         pm.get_end_time(), 60./change_bpm))
    assert np.allclose(pm.get_beats(), expected_beats)
    # When there are two time signature changes, make sure both get included
    pm.time_signature_changes.append(pretty_midi.TimeSignature(5, 4, 1.9))
    expected_beats[expected_beats == 2.] = 1.9
    assert np.allclose(pm.get_beats(), expected_beats)
    # Request a start time after time time signature change
    expected_beats = np.arange(2.2, change_time, 60./120.)
    expected_beats = np.append(
        expected_beats,
        change_time + (4.7 - change_time)/(60./120.)*60./change_bpm)
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 60./change_bpm,
                                         pm.get_end_time(), 60./change_bpm))
    assert np.allclose(pm.get_beats(2.2), expected_beats)


def test_get_downbeats():
    pm = pretty_midi.PrettyMIDI()
    # Add a note to force get_end_time() to be non-zero
    i = pretty_midi.Instrument(0)
    i.notes.append(pretty_midi.Note(100, 100, 0.3, 20.4))
    pm.instruments.append(i)
    # pretty_midi assumes 120 bpm, 4/4 unless otherwise specified
    assert np.allclose(pm.get_downbeats(),
                       np.arange(0, pm.get_end_time(), 4*60./120.))
    # Testing starting from a different beat time
    assert np.allclose(pm.get_downbeats(.2),
                       np.arange(0, pm.get_end_time(), 4*60./120.) + .2)
    # Testing a tempo change
    change_bpm = 93.
    change_time = 8.4
    pm._tick_scales.append(
        (pm.time_to_tick(change_time), 60./(change_bpm*pm.resolution)))
    pm._update_tick_to_time(pm.time_to_tick(pm.get_end_time()))
    # Track at 120 bpm up to the tempo change time
    expected_beats = np.arange(0, change_time, 4*60./120.)
    # BPM switches (4.5 - 4.4)/(60./120.) of the way through
    expected_beats = np.append(
        expected_beats,
        change_time + (10. - change_time)/(4*60./120.)*4*60./change_bpm)
    # From there, use the new bpm
    expected_beats = np.append(
        expected_beats, np.arange(expected_beats[-1] + 4*60./change_bpm,
                                  pm.get_end_time(), 4*60./change_bpm))
    assert np.allclose(pm.get_downbeats(), expected_beats)
    # When requesting a start_time after the tempo change, make sure we just
    # track as normal
    assert np.allclose(
        pm.get_downbeats(change_time + .1),
        np.arange(change_time + .1, pm.get_end_time(), 4*60./change_bpm))
    # Add a time signature change, which forces beat tracking to restart
    pm.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 2.1))
    # Track at 120 bpm up to time signature change
    expected_beats = np.arange(0, 2.1, 4*60./120.)
    # Now track, restarting from time signature change time
    expected_beats = np.append(expected_beats,
                               np.arange(2.1, change_time, 3*60./120.))
    # BPM switches (4.6 - 4.4)/(60./120.) of the way through
    expected_beats = np.append(
        expected_beats,
        change_time + (9.6 - change_time)/(3*60./120.)*3*60./change_bpm)
    # From there, use the new bpm
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 3*60./change_bpm,
                                         pm.get_end_time(), 3*60./change_bpm))
    assert np.allclose(pm.get_downbeats(), expected_beats)
    # When there are two time signature changes, make sure both get included
    pm.time_signature_changes.append(pretty_midi.TimeSignature(5, 4, 1.9))
    expected_beats[expected_beats == 2.] = 1.9
    assert np.allclose(pm.get_downbeats(), expected_beats)
    # Request a start time after time time signature change
    expected_beats = np.arange(2.2, change_time, 3*60./120.)
    expected_beats = np.append(
        expected_beats,
        change_time + (9.7 - change_time)/(3*60./120.)*3*60./change_bpm)
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 3*60./change_bpm,
                                         pm.get_end_time(), 3*60./change_bpm))
    assert np.allclose(pm.get_downbeats(2.2), expected_beats)
    # Test for compound meters
    pm = pretty_midi.PrettyMIDI()
    # Add a note to force get_end_time() to be non-zero
    i = pretty_midi.Instrument(0)
    i.notes.append(pretty_midi.Note(100, 100, 0.3, 20.4))
    pm.instruments.append(i)
    # Simple test, assume 6/8 time for the entire piece
    pm.time_signature_changes.append(pretty_midi.TimeSignature(6, 8, 0))
    assert np.allclose(pm.get_downbeats(),
                       np.arange(0, pm.get_end_time(), 3*60./120.))


def test_adjust_times():
    # Simple tests for adjusting note times
    def simple():
        pm = pretty_midi.PrettyMIDI()
        i = pretty_midi.Instrument(0)
        # Create 9 notes, at times [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for n, start in enumerate(range(1, 10)):
            i.notes.append(pretty_midi.Note(100, 100 + n, start, start + .5))
        pm.instruments.append(i)
        return pm
    # Test notes are interpolated as expected
    pm = simple()
    pm.adjust_times([0, 10], [5, 20])
    for note, start in zip(pm.instruments[0].notes, 1.5*np.arange(1, 10) + 5):
        assert note.start == start
    # Test notes are all ommitted when adjustment range doesn't cover them
    pm = simple()
    pm.adjust_times([10, 20], [5, 10])
    assert len(pm.instruments[0].notes) == 0
    # Test repeated mapping times
    pm = simple()
    pm.adjust_times([0, 5, 6.5, 10], [5, 10, 10, 17])
    # Original times  [1, 2, 3, 4,  7,  8,  9]
    # The notes at times 5 and 6 have their durations squashed to zero
    expected_starts = [6, 7, 8, 9, 11, 13, 15]
    assert np.allclose(
        [n.start for n in pm.instruments[0].notes], expected_starts)
    pm = simple()
    pm.adjust_times([0, 5, 5, 10], [7, 12, 13, 17])
    # Original times  [1, 2, 3, 4,  5,  6,  7,  8,  9]
    expected_starts = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert np.allclose(
        [n.start for n in pm.instruments[0].notes], expected_starts)

    # Complicated example
    pm = simple()
    # Include pitch bends and control changes to test adjust_events
    pm.instruments[0].pitch_bends.append(pretty_midi.PitchBend(100, 1.))
    # Include event which fall within omitted region
    pm.instruments[0].pitch_bends.append(pretty_midi.PitchBend(200, 7.))
    pm.instruments[0].pitch_bends.append(pretty_midi.PitchBend(0, 7.1))
    # Include event which falls outside of the track
    pm.instruments[0].pitch_bends.append(pretty_midi.PitchBend(10, 10.))
    pm.instruments[0].control_changes.append(
        pretty_midi.ControlChange(0, 0, .5))
    pm.instruments[0].control_changes.append(
        pretty_midi.ControlChange(0, 1, 5.5))
    pm.instruments[0].control_changes.append(
        pretty_midi.ControlChange(0, 2, 7.5))
    pm.instruments[0].control_changes.append(
        pretty_midi.ControlChange(0, 3, 20.))
    # Include track-level meta events to test adjust_meta
    pm.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, .1))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 5.2))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(6, 4, 6.2))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(5, 4, 15.3))
    pm.key_signature_changes.append(pretty_midi.KeySignature(1, 1.))
    pm.key_signature_changes.append(pretty_midi.KeySignature(2, 6.2))
    pm.key_signature_changes.append(pretty_midi.KeySignature(3, 7.2))
    pm.key_signature_changes.append(pretty_midi.KeySignature(4, 12.3))
    # Add in tempo changes - 100 bpm at 0s
    pm._tick_scales[0] = (0, 60./(100*pm.resolution))
    # 110 bpm at 6s
    pm._tick_scales.append((2200, 60./(110*pm.resolution)))
    # 120 bpm at 8.1s
    pm._tick_scales.append((3047, 60./(120*pm.resolution)))
    # 150 bpm at 8.3s
    pm._tick_scales.append((3135, 60./(150*pm.resolution)))
    # 80 bpm at 9.3s
    pm._tick_scales.append((3685, 60./(80*pm.resolution)))
    pm._update_tick_to_time(20000)

    # Adjust times, with a collapsing section in original and new times
    pm.adjust_times([2., 3.1, 3.1, 5.1, 7.5, 10],
                    [5., 6., 7., 8.5, 8.5, 11])

    # Original tempo change times: [0, 6, 8.1, 8.3, 9.3]
    # Plus tempo changes at each of new_times which are not collapsed
    # Plus tempo change at 0s by default
    expected_times = [0., 5., 6., 8.5,
                      8.5 + (6 - 5.1)*(11 - 8.5)/(10 - 5.1),
                      8.5 + (8.1 - 5.1)*(11 - 8.5)/(10 - 5.1),
                      8.5 + (8.3 - 5.1)*(11 - 8.5)/(10 - 5.1),
                      8.5 + (9.3 - 5.1)*(11 - 8.5)/(10 - 5.1)]
    # Tempos scaled by differences in timing, plus 120 bpm at the beginning
    expected_tempi = [120., 100*(3.1 - 2)/(6 - 5),
                      100*(5.1 - 3.1)/(8.5 - 6),
                      100*(10 - 5.1)/(11 - 8.5),
                      110*(10 - 5.1)/(11 - 8.5),
                      120*(10 - 5.1)/(11 - 8.5),
                      150*(10 - 5.1)/(11 - 8.5),
                      80*(10 - 5.1)/(11 - 8.5)]
    change_times, tempi = pm.get_tempo_changes()
    # Due to the fact that tempo change times must occur at discrete ticks, we
    # must raise the relative tolerance when comparing
    assert np.allclose(expected_times, change_times, rtol=.001)
    assert np.allclose(expected_tempi, tempi, rtol=.002)

    # Test that all other events were interpolated as expected
    note_starts = [5.0,
                   5 + 1/1.1,
                   6 + .9/(2/2.5),
                   6 + 1.9/(2/2.5),
                   8.5 + .5,
                   8.5 + 1.5]
    note_ends = [5 + .5/1.1,
                 6 + .4/(2/2.5),
                 6 + 1.4/(2/2.5),
                 8.5,
                 8.5 + 1.,
                 10 + .5]
    note_pitches = [101, 102, 103, 104, 107, 108]
    for note, s, e, p in zip(pm.instruments[0].notes, note_starts, note_ends,
                             note_pitches):
        assert note.start == s
        assert note.end == e
        assert note.pitch == p

    bend_times = [5., 8.5, 8.5]
    bend_pitches = [100, 200, 0]
    for bend, t, p in zip(pm.instruments[0].pitch_bends, bend_times,
                          bend_pitches):
        assert bend.time == t
        assert bend.pitch == p

    cc_times = [5., 8.5, 8.5]
    cc_values = [0, 1, 2]
    for cc, t, v in zip(pm.instruments[0].control_changes, cc_times,
                        cc_values):
        assert cc.time == t
        assert cc.value == v

    # The first time signature change will be placed at the first interpolated
    # downbeat location - so, start by computing the location of the first
    # downbeat after the start of original_times, then interpolate it
    first_downbeat_after = .1 + 2*3*60./100.
    first_ts_time = 6. + (first_downbeat_after - 3.1)/(2./2.5)
    ts_times = [first_ts_time, 8.5, 8.5]
    ts_numerators = [3, 4, 6]
    for ts, t, n in zip(pm.time_signature_changes, ts_times, ts_numerators):
        assert np.isclose(ts.time, t)
        assert ts.numerator == n

    ks_times = [5., 8.5, 8.5]
    ks_keys = [1, 2, 3]
    for ks, t, k in zip(pm.key_signature_changes, ks_times, ks_keys):
        assert ks.time == t
        assert ks.key_number == k


def test_properly_order_overlapping_notes():
    def make_mido_track(notes_str, file):
        track = mido.MidiTrack()
        for line in notes_str.split('\n'):
            line = line.strip()
            if line:
                track.append(mido.Message.from_str(line))
        mido_file = mido.MidiFile()
        mido_file.tracks.append(track)
        mido_file.save(file=file)

    # two notes with pitch 72 open at once
    bad_track = """
    note_on channel=0 note=72 velocity=88 time=0
    note_on channel=0 note=72 velocity=88 time=48
    note_on channel=0 note=72 velocity=0 time=0
    note_on channel=0 note=74 velocity=88 time=48
    note_on channel=0 note=72 velocity=0 time=0
    note_on channel=0 note=72 velocity=88 time=48
    note_on channel=0 note=74 velocity=0 time=0
    note_on channel=0 note=72 velocity=0 time=48
    """

    # the 72 note is first closed, then another one opened
    good_track = """
    note_on channel=0 note=72 velocity=88 time=0
    note_on channel=0 note=72 velocity=0 time=48
    note_on channel=0 note=72 velocity=88 time=0
    note_on channel=0 note=74 velocity=88 time=48
    note_on channel=0 note=72 velocity=0 time=0
    note_on channel=0 note=72 velocity=88 time=48
    note_on channel=0 note=74 velocity=0 time=0
    note_on channel=0 note=72 velocity=0 time=48
    """

    for kind, track in (('good', good_track), ('bad', bad_track)):
        with NamedTemporaryFile() as file:
            make_mido_track(track, file)
            file.seek(0)
            pm_song = pretty_midi.PrettyMIDI(file)

        def extract_notes(pm_track):
            return np.array([(note.pitch, note.end-note.start) for note
                             in pm_track.notes])

        expected = np.array([[72, 0.05], [72, 0.05], [74, 0.05], [72, 0.05]])
        assert np.allclose(expected, extract_notes(pm_song.instruments[0]))

        with NamedTemporaryFile() as file:
            pm_song.write(file)
            file.seek(0)
            pm_song_written = pretty_midi.PrettyMIDI(file)

        assert np.allclose(expected,
                           extract_notes(pm_song_written.instruments[0]))


def test_get_end_time():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)
    # When no events, end time should be 0
    assert pm.get_end_time() == 0
    # End time should be sensitive to inst notes, pitch bends, control changes
    inst.notes.append(
        pretty_midi.Note(start=0.5, end=1.7, pitch=30, velocity=100))
    assert np.allclose(pm.get_end_time(), 1.7)
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=100, time=1.9))
    assert np.allclose(pm.get_end_time(), 1.9)
    inst.control_changes.append(
        pretty_midi.ControlChange(number=0, value=10, time=2.1))
    assert np.allclose(pm.get_end_time(), 2.1)
    # End time should be sensitive to meta events
    pm.time_signature_changes.append(
        pretty_midi.TimeSignature(numerator=4, denominator=4, time=2.3))
    assert np.allclose(pm.get_end_time(), 2.3)
    pm.time_signature_changes.append(
        pretty_midi.KeySignature(key_number=10, time=2.5))
    assert np.allclose(pm.get_end_time(), 2.5)
    pm.time_signature_changes.append(pretty_midi.Lyric(text='hey', time=2.7))
    assert np.allclose(pm.get_end_time(), 2.7)


def test_estimate_tempi():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)
    # Add events corresponding to 120 bpm (0.5 seconds apart)
    for start in np.linspace(.5, 4.5, 5):
        inst.notes.append(pretty_midi.Note(pitch=40, velocity=100, start=start,
                                           end=start + .1))
    # Add events corresponding to 180 bpm (0.333... seconds apart)
    for start in np.linspace(0., 4., 13):
        inst.notes.append(pretty_midi.Note(pitch=40, velocity=100, start=start,
                                           end=start + .1))
    tempi, weights = pm.estimate_tempi()
    assert len(tempi) == 2
    assert np.allclose(tempi[0], 180.)
    assert np.allclose(tempi[1], 120.)
    assert weights[0] > weights[1]
    assert tempi[0] == pm.estimate_tempo()


def test_get_onsets():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)
    onsets = np.linspace(.5, 4.5, 5)
    for start in onsets:
        inst.notes.append(pretty_midi.Note(pitch=40, velocity=100, start=start,
                                           end=start + .1))
    assert np.allclose(pm.get_onsets(), onsets)


def test_get_piano_roll_and_get_chroma():
    pm = pretty_midi.PrettyMIDI()
    assert pm.get_piano_roll().shape == (128, 0)
    # Currently just a rudimentary test since it's hard to test things like
    # pitch bends correctly
    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)
    inst.notes.append(pretty_midi.Note(pitch=40, velocity=100, start=0.05,
                                       end=0.45))
    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)
    inst.notes.append(pretty_midi.Note(pitch=40, velocity=50, start=0.35,
                                       end=0.5))
    inst.notes.append(pretty_midi.Note(pitch=45, velocity=100, start=0.1,
                                       end=0.2))

    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)
    inst.control_changes.append(pretty_midi.ControlChange(number=64,
                                                          value=65,
                                                          time=0.12))
    inst.control_changes.append(pretty_midi.ControlChange(number=64,
                                                          value=63,
                                                          time=0.5))
    inst.notes.append(pretty_midi.Note(pitch=50, velocity=50, start=0.35,
                                       end=0.4))
    inst.notes.append(pretty_midi.Note(pitch=55, velocity=20, start=0.1,
                                       end=0.15))
    inst.notes.append(pretty_midi.Note(pitch=55, velocity=10, start=0.2,
                                       end=0.25))
    inst.notes.append(pretty_midi.Note(pitch=55, velocity=50, start=0.3,
                                       end=0.42))

    expected_piano_roll = np.zeros((128, 50))
    expected_piano_roll[40, 5:35] = 100
    expected_piano_roll[40, 35:45] = 150
    expected_piano_roll[40, 45:] = 50
    expected_piano_roll[45, 10:20] = 100
    expected_piano_roll[50, 35:40] = 50
    expected_piano_roll[55, 10:15] = 20
    expected_piano_roll[55, 20:25] = 10
    expected_piano_roll[55, 30:42] = 50
    assert np.allclose(pm.get_piano_roll(pedal_threshold=None),
                       expected_piano_roll)

    expected_piano_roll[50, 35:50] = 50
    expected_piano_roll[55, 10:30] = 20
    expected_piano_roll[55, 30:50] = 50
    assert np.allclose(pm.get_piano_roll(),
                       expected_piano_roll)

    expected_chroma = np.zeros((12, 50))
    expected_chroma[4, 5:35] = 100
    expected_chroma[4, 35:45] = 150
    expected_chroma[4, 45:] = 50
    expected_chroma[9, 10:20] = 100
    expected_chroma[2, 35:40] = 50
    expected_chroma[7, 10:15] = 20
    expected_chroma[7, 20:25] = 10
    expected_chroma[7, 30:42] = 50
    assert np.allclose(pm.get_chroma(pedal_threshold=None),
                       expected_chroma)

    expected_chroma[2, 35:50] = 50
    expected_chroma[7, 10:30] = 20
    expected_chroma[7, 30:50] = 50
    assert np.allclose(pm.get_chroma(),
                       expected_chroma)


def test_get_piano_roll_integrated():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)
    inst.notes.append(pretty_midi.Note(pitch=70, velocity=90, start=0.01,
                                       end=0.05))
    times = np.linspace(0., 0.06, num=12, endpoint=False)  # sampling at 200
    # piano roll samples at 100, needs to interpolate to 200
    piano_roll = pm.instruments[0].get_piano_roll(fs=100, times=times)
    expected_piano_roll = np.zeros((128, 12))
    expected_piano_roll[70, 2:10] = 90.
    assert np.allclose(piano_roll, expected_piano_roll)


def test_synthesize():
    pm = pretty_midi.PrettyMIDI()
    assert pm.synthesize().size == 0
    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)
    inst.notes.append(pretty_midi.Note(pitch=40, velocity=100, start=0.1,
                                       end=1.1))
    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)
    inst.notes.append(pretty_midi.Note(pitch=60, velocity=50, start=0.5,
                                       end=1.3))
    fs = 10000
    synthesized = pm.synthesize(fs=fs)
    # Synthesize pads 1 second of silence
    assert synthesized.shape[0] == (1.3 + 1)*fs
    # Should be normalied
    assert (np.allclose(synthesized.max(), 1) or
            np.allclose(synthesized.min(), -1))


def test_get_intervals_and_pitches():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    pm.instruments.append(inst)

    inst.notes.append(pretty_midi.Note(pitch=69, velocity=90, start=0.01,
                                       end=0.5))
    inst.notes.append(pretty_midi.Note(pitch=57, velocity=90, start=0.51,
                                       end=1.0))
    inst.notes.append(pretty_midi.Note(pitch=45, velocity=90, start=1.01,
                                       end=1.5))
    inst.notes.append(pretty_midi.Note(pitch=33, velocity=90, start=1.51,
                                       end=2.0))

    expected_intervals = np.array([[0.01, 0.5], [0.51, 1.0], [1.01, 1.5], [1.51, 2.0]])
    expected_pitches = np.array([440, 220, 110, 55])

    intervals, pitches = pm.get_intervals_and_pitches()

    assert np.allclose(intervals, expected_intervals)
    assert np.allclose(pitches, expected_pitches)


def test_metadata_handling():
    pm = pretty_midi.PrettyMIDI()
    pm.lyrics.append(pretty_midi.Lyric('word1', 0.5))
    pm.lyrics.append(pretty_midi.Lyric('word2', 1.0))
    pm.text_events.append(pretty_midi.Text('text1', 0.1))
    pm.text_events.append(pretty_midi.Text('text2', 1.5))

    assert pm.get_end_time() == 1.5

    # Test loading metadata from mido object
    mid = mido.MidiFile()
    track1 = mido.MidiTrack()
    track1.append(mido.MetaMessage('lyrics', text='l1', time=100))
    track1.append(mido.MetaMessage('text', text='t1', time=50))
    mid.tracks.append(track1)

    track2 = mido.MidiTrack()
    track2.append(mido.MetaMessage('lyrics', text='l2', time=50))
    track2.append(mido.MetaMessage('text', text='t2', time=150))
    mid.tracks.append(track2)

    pm_loaded = pretty_midi.PrettyMIDI(mido_object=mid)
    assert len(pm_loaded.lyrics) == 2
    assert pm_loaded.lyrics[0].text == 'l2'
    assert pm_loaded.lyrics[1].text == 'l1'
    assert len(pm_loaded.text_events) == 2
    assert pm_loaded.text_events[0].text == 't1'
    assert pm_loaded.text_events[1].text == 't2'


def test_instrument_loading_stragglers_and_names():
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name='TestTrack', time=0))
    # CC and PitchBend before note_on (stragglers)
    track.append(mido.Message('control_change', control=1, value=10, time=10))
    track.append(mido.Message('pitchwheel', pitch=1000, time=10))
    # Now a note_on
    track.append(mido.Message('note_on', note=60, velocity=64, time=10))
    track.append(mido.Message('note_off', note=60, velocity=64, time=10))
    mid.tracks.append(track)

    pm = pretty_midi.PrettyMIDI(mido_object=mid)
    assert len(pm.instruments) == 1
    inst = pm.instruments[0]
    assert inst.name == 'TestTrack'
    assert len(inst.control_changes) == 1
    assert inst.control_changes[0].number == 1
    assert len(inst.pitch_bends) == 1
    assert inst.pitch_bends[0].pitch == 1000
    assert len(inst.notes) == 1


def test_analysis_functions():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    # C4
    inst.notes.append(pretty_midi.Note(100, 60, 0, 1))
    # E4
    inst.notes.append(pretty_midi.Note(50, 64, 1, 3))
    pm.instruments.append(inst)

    # Histogram
    hist = pm.get_pitch_class_histogram()
    assert hist[0] == 0.5
    assert hist[4] == 0.5

    hist_dur = pm.get_pitch_class_histogram(use_duration=True)
    assert hist_dur[0] == 1/3.0
    assert hist_dur[4] == 2/3.0

    hist_vel = pm.get_pitch_class_histogram(use_velocity=True)
    assert hist_vel[0] == 100/150.0
    assert hist_vel[4] == 50/150.0

    # Transition Matrix
    # Add another note to have a transition
    inst.notes.append(pretty_midi.Note(100, 67, 3.01, 4)) # G4
    # Transitions: C4->E4 (0->4) and E4->G4 (4->7)
    trans = pm.get_pitch_class_transition_matrix(normalize=True)
    assert trans[4, 7] == 0.5
    assert trans[0, 4] == 0.5

    # Beat start estimation
    # Need more notes for beat start estimation
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for i in range(10):
        # Higher velocity for the first note to make 0.0 the best candidate
        v = 200 if i == 0 else 100
        inst.notes.append(pretty_midi.Note(v, 60, i * 0.5, i * 0.5 + 0.1))
    pm.instruments.append(inst)
    beat_start = pm.estimate_beat_start()
    assert any(np.isclose(beat_start, n.start) for n in inst.notes)

    # Test estimate_beat_start empty
    with pytest.raises(ValueError):
        pretty_midi.PrettyMIDI().estimate_beat_start()


def test_conversions_edge_cases():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    # tick_to_time
    assert pm.tick_to_time(0) == 0
    # Large tick
    with pytest.raises(IndexError):
        pm.tick_to_time(1e8)
    # Tick larger than currently mapped
    t = pm.tick_to_time(2000)
    assert t > 0
    # Non-int tick
    with pytest.warns(UserWarning, match='tick should be an int.'):
        pm.tick_to_time(10.5)

    # time_to_tick
    tick = pm.time_to_tick(10.0)
    assert tick > 0
    # Time beyond mapped range
    tick_far = pm.time_to_tick(1000.0)
    assert tick_far > tick


def test_synthesis_options():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 0, 1))
    pm.instruments.append(inst)

    # synthesize with normalize=False
    syn = pm.synthesize(normalize=False)
    # Should be small since it divides by len(inst) * 2**15
    assert syn.max() < 1.0

    # synthesize empty
    pm_empty = pretty_midi.PrettyMIDI()
    assert pm_empty.synthesize().size == 0

    # fluidsynth mock test
    from unittest.mock import MagicMock
    # We mock fluidsynth to avoid dependency issues during tests
    mock_synth = MagicMock()
    mock_synth.get_samples.return_value = np.zeros(1000)

    with MagicMock() as mock_get_fs:
        mock_get_fs.return_value = (mock_synth, 0, True)
        # We need to monkeypatch get_fluidsynth_instance
        # Use a local reference to avoid shadowing pretty_midi
        import pretty_midi.pretty_midi as pm_mod
        original_get_fs = pm_mod.get_fluidsynth_instance
        pm_mod.get_fluidsynth_instance = mock_get_fs

        # Mock instrument.fluidsynth because it's called inside pm.fluidsynth
        for instrument in pm.instruments:
            instrument.fluidsynth = MagicMock(return_value=np.zeros(1000))

        fs_syn = pm.fluidsynth()
        assert fs_syn.size == 1000

        # Restore
        pm_mod.get_fluidsynth_instance = original_get_fs


def test_cropping_and_invalid_notes():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 1, 2))
    inst.notes.append(pretty_midi.Note(100, 62, 3, 4))
    pm.instruments.append(inst)

    # crop
    pm.crop(0.5, 2.5)
    assert len(pm.instruments[0].notes) == 1
    assert pm.instruments[0].notes[0].pitch == 60

    # crop invalid
    with pytest.raises(ValueError, match='start_time must be non-negative.'):
        pm.crop(-1, 2)
    with pytest.raises(ValueError, match='end_time must be strictly higher than start_time.'):
        pm.crop(2, 1)

    # crop end_time > midi_end_time
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 1, 2))
    pm.instruments.append(inst)
    with pytest.warns(UserWarning, match='end_time is greater than the MIDI object\'s end time'):
        pm.crop(0, 5)

    # remove_invalid_notes
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    note = pretty_midi.Note(100, 60, 1, 2)
    note.end = 0.5 # Manually make it invalid
    inst.notes.append(note)
    pm.instruments.append(inst)
    pm.remove_invalid_notes()
    assert len(pm.instruments[0].notes) == 0


def test_write_and_read_back():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name='Test')
    inst.notes.append(pretty_midi.Note(100, 60, 0.5, 1.0))
    pm.instruments.append(inst)

    pm.lyrics.append(pretty_midi.Lyric('test lyric', 0.6))
    pm.key_signature_changes.append(pretty_midi.KeySignature(1, 0.2)) # G Major
    pm.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 0.1))

    with NamedTemporaryFile() as f:
        pm.write(f.name)
        pm_read = pretty_midi.PrettyMIDI(f.name)

    assert len(pm_read.instruments) == 1
    assert pm_read.instruments[0].name == 'Test'
    assert len(pm_read.lyrics) == 1
    assert pm_read.lyrics[0].text == 'test lyric'
    assert len(pm_read.key_signature_changes) == 1
    assert pm_read.key_signature_changes[0].key_number == 1
    # Note: PrettyMIDI adds a default 4/4 at 0 if none exists,
    # or it might have changed based on our 3/4 at 0.1
    assert any(ts.numerator == 3 for ts in pm_read.time_signature_changes)


def test_corrupt_midi_init():
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.Message('note_on', note=60, time=int(2e7))) # Very large time
    mid.tracks.append(track)
    with pytest.raises(ValueError, match='it is likely corrupt'):
        pretty_midi.PrettyMIDI(mido_object=mid)


def test_tempo_signature_warnings():
    mid = mido.MidiFile()
    track0 = mido.MidiTrack()
    track0.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    mid.tracks.append(track0)

    track1 = mido.MidiTrack()
    # Tempo change on track 1 (invalid for type 1)
    track1.append(mido.MetaMessage('set_tempo', tempo=400000, time=100))
    mid.tracks.append(track1)

    with pytest.warns(RuntimeWarning, match='Tempo, Key or Time signature change events found on non-zero tracks'):
        pretty_midi.PrettyMIDI(mido_object=mid)


def test_instrument_coverage():
    # is_drum histogram
    inst = pretty_midi.Instrument(0, is_drum=True)
    inst.notes.append(pretty_midi.Note(100, 60, 0, 1))
    assert np.all(inst.get_pitch_class_histogram() == 0)

    # get_piano_roll with sustain pedal
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 0, 0.5))
    # Sustain pedal on at 0.4, off at 1.0
    inst.control_changes.append(pretty_midi.ControlChange(64, 100, 0.4))
    inst.control_changes.append(pretty_midi.ControlChange(64, 0, 1.0))
    pr = inst.get_piano_roll(fs=100)
    # Note should be extended to 1.0
    assert np.all(pr[60, 50:100] == 100)

    # get_piano_roll with pitch bend
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 0, 1.0))
    # Pitch bend +1 semitone at 0.5
    inst.pitch_bends.append(pretty_midi.PitchBend(4096, 0.5)) # 4096/8192 * 2 = 1 semitone
    pr = inst.get_piano_roll(fs=100)
    # Check that some energy moved to 61
    assert np.any(pr[61, 50:100] > 0)

    # get_piano_roll drum track empty
    inst = pretty_midi.Instrument(0, is_drum=True)
    assert inst.get_piano_roll().shape[0] == 128
    assert np.all(inst.get_piano_roll(times=np.array([0, 1])) == 0)

    # synthesize invalid wave
    inst_not_drum = pretty_midi.Instrument(0, is_drum=False)
    # Need a note to avoid early return if get_end_time is 0
    inst_not_drum.notes.append(pretty_midi.Note(100, 60, 0, 1))
    with pytest.raises(ValueError, match='wave should be a callable'):
        inst_not_drum.synthesize(wave='not callable')

    # fluidsynth with more events
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 0, 1))
    inst.pitch_bends.append(pretty_midi.PitchBend(1000, 0.5))
    inst.control_changes.append(pretty_midi.ControlChange(1, 100, 0.2))
    pm.instruments.append(inst)

    from unittest.mock import MagicMock
    mock_synth = MagicMock()
    # Return 2*n samples because the code does [::2]
    mock_synth.get_samples.side_effect = lambda n: np.zeros(2 * n)
    mock_synth.get_setting.return_value = 44100

    import pretty_midi.instrument as inst_mod
    import pretty_midi.pretty_midi as pm_mod
    original_get_fs_inst = inst_mod.get_fluidsynth_instance
    original_get_fs_pm = pm_mod.get_fluidsynth_instance
    with MagicMock() as mock_get_fs:
        mock_get_fs.return_value = (mock_synth, 0, True)
        inst_mod.get_fluidsynth_instance = mock_get_fs
        pm_mod.get_fluidsynth_instance = mock_get_fs
        
        # We need to test the fluidsynth method of the instrument specifically
        # to hit its internal loops
        syn = inst.fluidsynth(synthesizer=mock_synth)
        assert syn.size > 0
        assert mock_synth.noteon.called
        assert mock_synth.pitch_bend.called
        assert mock_synth.cc.called

    inst_mod.get_fluidsynth_instance = original_get_fs_inst
    pm_mod.get_fluidsynth_instance = original_get_fs_pm


def test_note_repr():
    note = pretty_midi.Note(100, 60, 0.5, 1.0)
    assert 'Note' in repr(note)
    assert note.duration == 0.5


def test_containers_repr_str():
    # PitchBend
    pb = pretty_midi.PitchBend(100, 0.5)
    assert 'PitchBend' in repr(pb)
    # ControlChange
    cc = pretty_midi.ControlChange(1, 100, 0.5)
    assert 'ControlChange' in repr(cc)
    # TimeSignature
    ts = pretty_midi.TimeSignature(4, 4, 0.5)
    assert 'TimeSignature' in repr(ts)
    assert '4/4' in str(ts)
    with pytest.raises(ValueError):
        pretty_midi.TimeSignature(0, 4, 0.5)
    with pytest.raises(ValueError):
        pretty_midi.TimeSignature(4, 0, 0.5)
    with pytest.raises(ValueError):
        pretty_midi.TimeSignature(4, 4, -1)
    # KeySignature
    ks = pretty_midi.KeySignature(0, 0.5)
    assert 'KeySignature' in repr(ks)
    assert 'C Major' in str(ks)
    with pytest.raises(ValueError):
        pretty_midi.KeySignature(24, 0.5)
    with pytest.raises(ValueError):
        pretty_midi.KeySignature(0, -1)
    # Lyric
    lyric = pretty_midi.Lyric('test', 0.5)
    assert 'Lyric' in repr(lyric)
    assert 'test' in str(lyric)
    # Text
    text = pretty_midi.Text('test', 0.5)
    assert 'Text' in repr(text)
    assert 'test' in str(text)


def test_pretty_midi_repr():
    pm = pretty_midi.PrettyMIDI()
    assert 'PrettyMIDI' in repr(pm)
    inst = pretty_midi.Instrument(0, name='Test')
    assert 'Instrument' in repr(inst)
