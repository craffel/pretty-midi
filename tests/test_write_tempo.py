import io

import mido
import pretty_midi


def _write_midi(pm, **kwargs):
    output = io.BytesIO()
    pm.write(output, **kwargs)
    output.seek(0)
    return mido.MidiFile(file=output)


def _tempo_events(midi):
    tick = 0
    events = []
    for event in midi.tracks[0]:
        tick += event.time
        if event.type == 'set_tempo':
            events.append((tick, event.tempo))
    return events


def test_write_rounds_initial_tempo():
    for bpm, tempo in [(90, 666667), (123, 487805), (137, 437956),
                       (120, 500000), (240, 250000),
                       (6e7 / 500000.4, 500000), (6e7 / 500000.6, 500001)]:
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        assert _tempo_events(_write_midi(pm, round_tempo=True)) == [(0, tempo)]


def test_write_truncates_tempo_by_default():
    for bpm, tempo in [(90, 666666), (123, 487804), (137, 437956),
                       (120, 500000), (240, 250000),
                       (6e7 / 500000.4, 500000), (6e7 / 500000.6, 500000)]:
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        assert _tempo_events(_write_midi(pm)) == [(0, tempo)]
        written = _write_midi(pm, round_tempo=False)
        assert _tempo_events(written) == [(0, tempo)]


def test_write_preserves_integer_tempo_changes():
    for resolution in [96, 220, 480, 960]:
        source = mido.MidiFile(ticks_per_beat=resolution)
        tempos = [500000, 437956, 999999, 487805, 1, 16777215]
        source.tracks.append(mido.MidiTrack([
            mido.MetaMessage('set_tempo', tempo=tempo,
                             time=resolution if index else 0)
            for index, tempo in enumerate(tempos)]))
        source.tracks.append(mido.MidiTrack([
            mido.Message('note_on', note=60, velocity=100),
            mido.Message('note_off', note=60, time=6 * resolution)]))
        expected = [(index * resolution, tempo)
                    for index, tempo in enumerate(tempos)]
        for _ in range(3):
            encoded = io.BytesIO()
            source.save(file=encoded)
            encoded.seek(0)
            pm = pretty_midi.PrettyMIDI(encoded)
            source = _write_midi(pm, round_tempo=True)
            assert source.ticks_per_beat == resolution
            assert _tempo_events(source) == expected
            notes = [(event.time, event.note, event.velocity)
                     for event in source.tracks[1]
                     if event.type == 'note_on']
            assert notes == [(0, 60, 100), (6 * resolution, 60, 0)]


def test_write_keeps_note_ticks_with_rounded_tempo():
    pm = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=123)
    instrument = pretty_midi.Instrument(0)
    instrument.notes.append(pretty_midi.Note(100, 60, 0, 60. / 123))
    pm.instruments.append(instrument)
    original_scales = list(pm._tick_scales)
    written = _write_midi(pm, round_tempo=True)
    assert _tempo_events(written) == [(0, 487805)]
    tick = 0
    notes = []
    for event in written.tracks[1]:
        tick += event.time
        if event.type == 'note_on':
            notes.append((tick, event.note, event.velocity))
    assert notes == [(0, 60, 100), (480, 60, 0)]
    assert pm._tick_scales == original_scales
    assert instrument.notes[0].end == 60. / 123
