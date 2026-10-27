import numpy as np
import pytest

import pretty_midi


@pytest.mark.parametrize('kind', ['empty', 'drums', 'zero_velocity'])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('wave', [np.sin, np.cos])
def test_synthesize_silence(kind, normalize, wave):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0, is_drum=(kind == 'drums'))
    if kind != 'empty':
        velocity = 0 if kind == 'zero_velocity' else 100
        instrument.notes.append(pretty_midi.Note(velocity, 60, 0., .2))
    midi.instruments.append(instrument)

    with np.errstate(divide='raise', invalid='raise'):
        audio = midi.synthesize(fs=1000, wave=wave, normalize=normalize)

    expected_length = 1000 if kind == 'empty' else 1200
    np.testing.assert_array_equal(audio, np.zeros(expected_length))
