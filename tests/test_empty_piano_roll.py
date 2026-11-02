import numpy as np
import pytest

import pretty_midi


@pytest.mark.parametrize('kind', [
    'instrument', 'drums', 'score', 'empty_track'])
@pytest.mark.parametrize('times', [
    None, np.array([]), np.array([0.]), np.array([0., .1, .3, 1.])])
def test_empty_piano_roll_preserves_time_grid(kind, times):
    if kind in ('instrument', 'drums'):
        obj = pretty_midi.Instrument(0, is_drum=(kind == 'drums'))
        obj.control_changes.append(pretty_midi.ControlChange(64, 127, .5))
    else:
        obj = pretty_midi.PrettyMIDI()
        if kind == 'empty_track':
            obj.instruments.append(pretty_midi.Instrument(0))

    columns = 0 if times is None else times.shape[0]
    np.testing.assert_array_equal(
        obj.get_piano_roll(times=times), np.zeros((128, columns)))
    np.testing.assert_array_equal(
        obj.get_chroma(times=times), np.zeros((12, columns)))
