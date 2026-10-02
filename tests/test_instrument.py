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
