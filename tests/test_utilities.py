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
