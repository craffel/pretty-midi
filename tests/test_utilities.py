import pretty_midi
import pytest


def test_key_name_to_key_number():
    # First, test that number->name->number works
    for key_number in range(24):
        assert pretty_midi.key_name_to_key_number(
            pretty_midi.key_number_to_key_name(key_number)) == key_number
    # Explicitly test all valid input
    key_pc = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}
    for key in key_pc:
        for flatsharp, shift in zip(['', '#', 'b'], [0, 1, -1]):
            key_number = (key_pc[key] + shift) % 12
            for space in [' ', '']:
                for mode in ['M', 'Maj', 'Major', 'maj', 'major']:
                    assert pretty_midi.key_name_to_key_number(
                        key + flatsharp + space + mode) == key_number
                # Also ensure uppercase key name plus no mode string
                assert pretty_midi.key_name_to_key_number(
                    key.upper() + flatsharp + space) == key_number
                for mode in ['m', 'Min', 'Minor', 'min', 'minor']:
                    assert pretty_midi.key_name_to_key_number(
                        key + flatsharp + space + mode) == key_number + 12
                assert pretty_midi.key_name_to_key_number(
                    key + flatsharp + space) == key_number + 12
    # Test some invalid inputs
    for invalid_key in ['C#  m', 'C# ma', 'ba', 'bm m', 'f## Major', 'O']:
        with pytest.raises(ValueError):
            pretty_midi.key_name_to_key_number(invalid_key)


def test_qpm_to_bpm():
    # Test that twice the qpm leads to double the bpm for a range of qpm
    for qpm in [60, 100, 125.56]:
        for num in range(1, 24):
                for den in range(1, 64):
                        assert 2 * pretty_midi.qpm_to_bpm(qpm, num, den) \
                            == pretty_midi.qpm_to_bpm(qpm * 2, num, den)
    # Test that twice the denominator leads to double the bpm for a range
    # of denominators (those outside of this set just fall back to the
    # default of returning qpm.
    for qpm in [60, 100, 125.56]:
        for num in range(1, 24):
                for den in [1, 2, 4, 8, 16]:
                        assert 2 * pretty_midi.qpm_to_bpm(qpm, num, den) \
                            == pretty_midi.qpm_to_bpm(qpm, num, den * 2)
    # Check all compound meters
    # qpb is quarter notes per beat. qpm / qpb = q/m / q/b = b/m = bpm
    for den, qpb in zip([1, 2, 4, 8, 16, 32],
                        [12.0, 6.0, 3.0, 3/2.0, 3/4.0, 3/8.0]):
        for qpm in [60, 120, 125.56]:
            for num in range(2 * 3, 8 * 3, 3):
                assert pretty_midi.qpm_to_bpm(qpm, num, den) == qpm / qpb
    # Check all simple meters
    # qpb is quarter notes per beat. qpm / qpb = q/m / q/b = b/m = bpm
    for den, qpb in zip([1, 2, 4, 8, 16, 32],
                        [4.0, 2.0, 1.0, 1/2.0, 1/4.0, 1/8.0]):
        for qpm in [60, 120, 125.56]:
            for num in range(1, 24):
                if num > 3 and num % 3 == 0:
                    continue
                assert pretty_midi.qpm_to_bpm(qpm, num, den) == qpm / qpb
    # Test invalid inputs
    den = 4
    num = 4
    for qpm in [-1, 0, 'invalid']:
        with pytest.raises(ValueError):
            pretty_midi.qpm_to_bpm(qpm, num, den)
    qpm = 120
    for num in [-1, 0, 4.3, 'invalid']:
        with pytest.raises(ValueError):
            pretty_midi.qpm_to_bpm(qpm, num, den)
    num = 4
    for den in [-1, 0, 4.3, 'invalid']:
        with pytest.raises(ValueError):
            pretty_midi.qpm_to_bpm(qpm, num, den)
