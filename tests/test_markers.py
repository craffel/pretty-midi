"""Regression coverage for MIDI marker metadata."""
from io import BytesIO

import mido
import numpy as np
import pretty_midi


def _write_read(midi):
    output = BytesIO()
    midi.write(output)
    output.seek(0)
    return mido.MidiFile(file=output, charset=midi._charset)


def _marker_ticks(midi):
    markers = []
    for track in midi.tracks:
        tick = 0
        for event in track:
            tick += event.time
            if event.type == 'marker':
                markers.append((event.text, tick))
    return markers


def test_marker_read_write():
    source = mido.MidiFile(ticks_per_beat=480)
    source.tracks.append(mido.MidiTrack([
        mido.MetaMessage('set_tempo', tempo=500000, time=0),
        mido.MetaMessage('marker', text='Intro', time=0),
        mido.MetaMessage('set_tempo', tempo=1000000, time=480),
        mido.MetaMessage('marker', text='Chorus', time=480),
    ]))
    source_bytes = BytesIO()
    source.save(file=source_bytes)
    source_bytes.seek(0)
    midi = pretty_midi.PrettyMIDI(source_bytes)
    decoded = _write_read(midi)
    assert _marker_ticks(decoded) == [('Intro', 0), ('Chorus', 960)]
    assert [marker.text for marker in midi.markers] == ['Intro', 'Chorus']
    np.testing.assert_allclose([m.time for m in midi.markers], [0, 1.5])
    assert midi.get_end_time() == 1.5


def test_markers_merge_all_tracks():
    source = mido.MidiFile(ticks_per_beat=480)
    source.tracks.append(mido.MidiTrack([
        mido.MetaMessage('marker', text='First', time=480),
        mido.MetaMessage('marker', text='Second', time=0),
    ]))
    source.tracks.append(mido.MidiTrack([
        mido.MetaMessage('marker', text='Intro', time=0),
        mido.MetaMessage('text', text='Text', time=480),
        mido.MetaMessage('lyrics', text='Lyric', time=0),
        mido.MetaMessage('marker', text='Third', time=0),
        mido.MetaMessage('marker', text='', time=480),
    ]))
    midi = pretty_midi.PrettyMIDI(mido_object=source)
    assert [(m.text, m.time) for m in midi.markers] == [
        ('Intro', 0), ('First', 0.5), ('Second', 0.5), ('Third', 0.5), ('', 1)]
    assert [(e.text, e.time) for e in midi.text_events] == [('Text', 0.5)]
    assert [(e.text, e.time) for e in midi.lyrics] == [('Lyric', 0.5)]
    decoded = _write_read(midi)
    assert len(decoded.tracks) == 1
    assert _marker_ticks(decoded) == [
        ('Intro', 0), ('First', 480), ('Second', 480),
        ('Third', 480), ('', 960)]


def test_marker_charset_round_trip():
    source = mido.MidiFile(type=0, charset='utf-8', ticks_per_beat=480)
    source.tracks.append(mido.MidiTrack([
        mido.MetaMessage('marker', text='\u526f\u6b4c', time=480),
    ]))
    data = BytesIO()
    source.save(file=data)
    # The explicit charset also applies to pre-loaded mido objects.
    loaded = pretty_midi.PrettyMIDI(mido_object=source, charset=source.charset)
    assert _marker_ticks(_write_read(loaded)) == [('\u526f\u6b4c', 480)]
    for _ in range(3):
        data.seek(0)
        midi = pretty_midi.PrettyMIDI(data, charset='utf-8')
        assert [(m.text, m.time) for m in midi.markers] == [
            ('\u526f\u6b4c', 0.5)]
        decoded = _write_read(midi)
        assert _marker_ticks(decoded) == [('\u526f\u6b4c', 480)]
        data = BytesIO()
        decoded.save(file=data)


def test_marker_container_and_empty_initialization():
    marker = pretty_midi.Marker('The "chorus"', 1.25)
    assert marker.text == 'The "chorus"'
    assert marker.time == 1.25
    assert repr(marker) == 'Marker(text="The \\"chorus\\"", time=1.25)'
    assert str(marker) == '"The "chorus"" at 1.25 seconds'
    midi = pretty_midi.PrettyMIDI()
    assert midi.markers == []
    assert midi.get_end_time() == 0
    assert _marker_ticks(_write_read(midi)) == []
    decoded = pretty_midi.PrettyMIDI(mido_object=_write_read(midi))
    assert decoded.markers == []


def test_write_markers_from_scratch():
    midi = pretty_midi.PrettyMIDI(resolution=480)
    late = pretty_midi.Marker('Outro', 2)
    early = pretty_midi.Marker('Intro', 0)
    # These distinct times quantize to the same tick. Keep chronological order.
    second = pretty_midi.Marker('Second', 1.0002)
    first = pretty_midi.Marker('First', 1.0001)
    simultaneous = pretty_midi.Marker('Also first', 1.0001)
    midi.markers = [late, second, early, first, simultaneous]
    order = list(midi.markers)
    expected = [('Intro', 0), ('First', 960), ('Also first', 960),
                ('Second', 960), ('Outro', 1920)]
    assert _marker_ticks(_write_read(midi)) == expected
    assert _marker_ticks(_write_read(midi)) == expected
    assert midi.markers == order
    assert [m.time for m in midi.markers] == [
        2, 1.0002, 0, 1.0001, 1.0001]
    assert midi.get_end_time() == 2


def test_adjust_marker_times():
    midi = pretty_midi.PrettyMIDI()
    midi.markers = [pretty_midi.Marker(text, time) for text, time in [
        ('after', 5), ('middle', 3), ('before', 0.5), ('start', 1),
        ('same start', 1), ('knot', 2), ('end', 4)]]
    original_list = midi.markers
    midi.adjust_times([1, 2, 4], [2, 3, 7])
    assert midi.markers is original_list
    assert [m.text for m in midi.markers] == [
        'start', 'same start', 'knot', 'middle', 'end']
    np.testing.assert_allclose([m.time for m in midi.markers], [2, 2, 3, 5, 7])
    assert midi.get_end_time() == 7
    assert [text for text, _ in _marker_ticks(_write_read(midi))] == [
        'start', 'same start', 'knot', 'middle', 'end']


def test_adjust_markers_collapsed_times():
    midi = pretty_midi.PrettyMIDI()
    midi.markers = [pretty_midi.Marker(text, time) for text, time in [
        ('first', 0), ('second', 1), ('third', 2), ('last', 3)]]
    midi.adjust_times([0, 1, 2, 3], [0, 0, 0, 1])
    assert [(m.text, m.time) for m in midi.markers] == [
        ('first', 0), ('second', 0), ('third', 0), ('last', 1)]


def test_crop_markers():
    midi = pretty_midi.PrettyMIDI()
    midi.markers = [pretty_midi.Marker(text, time) for text, time in [
        ('before', 0), ('start', 1), ('same start', 1), ('middle', 2),
        ('end', 3), ('after', 4)]]
    midi.crop(1, 3)
    assert [(m.text, m.time) for m in midi.markers] == [
        ('start', 1), ('same start', 1), ('middle', 2), ('end', 3)]
    assert midi.get_end_time() == 3
    # The default crop must retain the final marker defining the end time.
    midi.crop()
    assert [(m.text, m.time) for m in midi.markers] == [
        ('start', 1), ('same start', 1), ('middle', 2), ('end', 3)]


def test_crop_discards_outside_markers():
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    instrument.notes.append(pretty_midi.Note(100, 60, 1, 2))
    midi.instruments.append(instrument)
    midi.markers = [pretty_midi.Marker('before', 0),
                    pretty_midi.Marker('after', 4)]
    midi.crop(1, 3)
    assert midi.markers == []
    assert len(midi.instruments[0].notes) == 1


def test_retime_markers_with_notes_and_tempo_changes():
    source = mido.MidiFile(ticks_per_beat=480)
    source.tracks.append(mido.MidiTrack([
        mido.MetaMessage('set_tempo', tempo=500000, time=0),
        mido.MetaMessage('set_tempo', tempo=1000000, time=480),
        mido.MetaMessage('marker', text='Note start', time=480),
        mido.MetaMessage('marker', text='Note end', time=480),
    ]))
    source.tracks.append(mido.MidiTrack([
        mido.Message('note_on', note=60, velocity=100, time=960),
        mido.Message('note_off', note=60, time=480),
    ]))
    midi = pretty_midi.PrettyMIDI(mido_object=source)
    midi.adjust_times([0, 0.5, 2.5], [0, 1, 2])
    note = midi.instruments[0].notes[0]
    assert (note.start, note.end) == (1.5, 2)
    assert [m.time for m in midi.markers] == [note.start, note.end]
    decoded = _write_read(midi)
    note_ticks = []
    tick = 0
    for event in decoded.tracks[1]:
        tick += event.time
        if event.type == 'note_on':
            note_ticks.append(tick)
    assert [tick for _, tick in _marker_ticks(decoded)] == note_ticks


def test_many_markers_in_example_file():
    midi = pretty_midi.PrettyMIDI('example.mid')
    end = midi.get_end_time()
    midi.markers = [pretty_midi.Marker(str(i), end * i / 1024)
                    for i in range(1025)]
    midi.markers.append(pretty_midi.Marker('Same position', end / 2))
    midi.adjust_times([0, end], [0, end * 2])
    expected = [(m.text, midi.time_to_tick(m.time)) for m in
                sorted(midi.markers, key=lambda m: m.time)]
    decoded = _write_read(midi)
    assert _marker_ticks(decoded) == expected
    output = BytesIO()
    decoded.save(file=output)
    output.seek(0)
    reloaded = pretty_midi.PrettyMIDI(output)
    assert [m.text for m in reloaded.markers] == [text for text, _ in expected]
