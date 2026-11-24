import io

import mido
import pytest

import pretty_midi


def _write(score, **kwargs):
    output = io.BytesIO()
    score.write(output, **kwargs)
    output.seek(0)
    return mido.MidiFile(file=output)


def test_type_zero_merges_at_absolute_ticks():
    source = mido.MidiFile(ticks_per_beat=96)
    source.tracks.append(mido.MidiTrack([
        mido.MetaMessage('set_tempo', tempo=1000000, time=96)]))
    score = pretty_midi.PrettyMIDI(mido_object=source)
    piano = pretty_midi.Instrument(0, name='Piano')
    piano.notes.append(pretty_midi.Note(90, 60, 0, 1.5))
    drums = pretty_midi.Instrument(0, is_drum=True, name='Drums')
    drums.notes.append(pretty_midi.Note(80, 36, 0.25, 0.5))
    score.instruments = [piano, drums]
    score.lyrics.append(pretty_midi.Lyric('end', 2.5))

    written = _write(score, midi_type=0)
    assert (written.type, len(written.tracks), written.ticks_per_beat) == (
        0, 1, 96)
    track = written.tracks[0]
    tick = 0
    notes = []
    tempos = []
    for msg in track:
        tick += msg.time
        if msg.type == 'note_on':
            notes.append((tick, msg.channel, msg.note, msg.velocity))
        elif msg.type == 'set_tempo':
            tempos.append((tick, msg.tempo))
    assert tempos == [(0, 500000), (96, 1000000)]
    assert notes == [(0, 0, 60, 90), (48, 9, 36, 80),
                     (96, 9, 36, 0), (192, 0, 60, 0)]
    assert [(m.type, m.time) for m in track[-2:]] == [
        ('lyrics', 96), ('end_of_track', 1)]
    assert sum(m.type == 'end_of_track' for m in track) == 1
    assert not any(m.type == 'track_name' for m in track)

    default = _write(score)
    assert default.type == 1
    explicit = _write(score, midi_type=1)
    assert explicit.type == 1
    assert default.tracks == explicit.tracks
    assert [t.name for t in default.tracks] == ['', 'Piano', 'Drums']


def test_type_zero_uses_all_available_channels():
    score = pretty_midi.PrettyMIDI()
    score.instruments = [pretty_midi.Instrument(i) for i in range(15)]
    score.instruments.insert(1, pretty_midi.Instrument(0, is_drum=True))
    changes = [m for m in _write(score, midi_type=0).tracks[0]
               if m.type == 'program_change']
    assert len(changes) == 16
    assert set(m.channel for m in changes) == set(range(16))
    assert changes[1].channel == 9

    for instruments in ([pretty_midi.Instrument(i) for i in range(16)],
                        [pretty_midi.Instrument(0, is_drum=True)
                         for _ in range(2)]):
        score.instruments = instruments
        with pytest.raises(ValueError):
            _write(score, midi_type=0)
        assert len(_write(score).tracks) == len(instruments) + 1


def test_type_zero_single_track_name():
    score = pretty_midi.PrettyMIDI()
    assert len(_write(score, midi_type=0).tracks) == 1
    score.instruments.append(pretty_midi.Instrument(1, name='Piano'))
    assert _write(score, midi_type=0).tracks[0].name == 'Piano'


def test_write_rejects_unsupported_type():
    with pytest.raises(ValueError):
        _write(pretty_midi.PrettyMIDI(), midi_type=2)
