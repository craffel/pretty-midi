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


def test_key_number_to_key_name():
    # Test major keys
    assert pretty_midi.key_number_to_key_name(0) == 'C Major'
    assert pretty_midi.key_number_to_key_name(1) == 'Db Major'
    # Test minor keys with sharps (preference keys)
    assert pretty_midi.key_number_to_key_name(13) == 'C# minor'
    assert pretty_midi.key_number_to_key_name(18) == 'F# minor'
    assert pretty_midi.key_number_to_key_name(20) == 'G# minor'
    # Test other minor keys
    assert pretty_midi.key_number_to_key_name(12) == 'C minor'
    assert pretty_midi.key_number_to_key_name(14) == 'D minor'

    # Test invalid inputs
    with pytest.raises(ValueError):
        pretty_midi.key_number_to_key_name(1.5)
    with pytest.raises(ValueError):
        pretty_midi.key_number_to_key_name(-1)
    with pytest.raises(ValueError):
        pretty_midi.key_number_to_key_name(24)


def test_mode_accidentals_to_key_number():
    # Test major keys with sharps
    assert pretty_midi.mode_accidentals_to_key_number(0, 0) == 0  # C Major
    assert pretty_midi.mode_accidentals_to_key_number(0, 1) == 7  # G Major
    assert pretty_midi.mode_accidentals_to_key_number(0, 6) == 6  # F# Major
    # Test major keys with flats
    assert pretty_midi.mode_accidentals_to_key_number(0, -1) == 5  # F Major
    assert pretty_midi.mode_accidentals_to_key_number(0, -2) == 10  # Bb Major
    assert pretty_midi.mode_accidentals_to_key_number(0, -7) == 11  # Cb Major -> B Major
    # Test minor keys
    assert pretty_midi.mode_accidentals_to_key_number(1, 0) == 21  # A minor
    assert pretty_midi.mode_accidentals_to_key_number(1, 1) == 16  # E minor

    # Test invalid accidentals
    for invalid_acc in [-8, 8, 1.5, 'invalid']:
        with pytest.raises(ValueError):
            pretty_midi.mode_accidentals_to_key_number(0, invalid_acc)
    # Test invalid mode
    for invalid_mode in [-1, 2, 0.5, 'invalid']:
        with pytest.raises(ValueError):
            pretty_midi.mode_accidentals_to_key_number(invalid_mode, 0)


def test_key_number_to_mode_accidentals():
    # Test major keys
    assert pretty_midi.key_number_to_mode_accidentals(0) == (0, 0)  # C Major
    assert pretty_midi.key_number_to_mode_accidentals(7) == (0, 1)  # G Major
    # Test minor keys
    assert pretty_midi.key_number_to_mode_accidentals(21) == (1, 0)  # A minor
    assert pretty_midi.key_number_to_mode_accidentals(16) == (1, 1)  # E minor

    # Test invalid inputs
    with pytest.raises(ValueError):
        pretty_midi.key_number_to_mode_accidentals(24)
    with pytest.raises(ValueError):
        pretty_midi.key_number_to_mode_accidentals(-1)
    with pytest.raises(ValueError):
        pretty_midi.key_number_to_mode_accidentals(1.5)


def test_note_frequency_conversion():
    # Test note_number_to_hz
    assert pretty_midi.note_number_to_hz(69) == 440.0
    assert pretty_midi.note_number_to_hz(60) == 440.0 * (2.0**(-9/12.0))
    # Test hz_to_note_number
    import numpy as np
    assert np.allclose(pretty_midi.hz_to_note_number(440.0), 69.0)
    assert np.allclose(pretty_midi.hz_to_note_number(pretty_midi.note_number_to_hz(60)), 60.0)


def test_note_name_to_number():
    assert pretty_midi.note_name_to_number('C4') == 60
    assert pretty_midi.note_name_to_number('A4') == 69
    assert pretty_midi.note_name_to_number('C#4') == 61
    assert pretty_midi.note_name_to_number('Cb4') == 59
    assert pretty_midi.note_name_to_number('C!4') == 59
    assert pretty_midi.note_name_to_number('c4') == 60
    assert pretty_midi.note_name_to_number('C-1') == 0
    # Test invalid format
    with pytest.raises(ValueError):
        pretty_midi.note_name_to_number('invalid')
    with pytest.raises(ValueError):
        pretty_midi.note_name_to_number('C')


def test_note_number_to_name():
    assert pretty_midi.note_number_to_name(60) == 'C4'
    assert pretty_midi.note_number_to_name(61) == 'C#4'
    assert pretty_midi.note_number_to_name(69) == 'A4'
    assert pretty_midi.note_number_to_name(60.1) == 'C4'


def test_drum_name_conversions():
    # Test note_number_to_drum_name
    assert pretty_midi.note_number_to_drum_name(35) == 'Acoustic Bass Drum'
    assert pretty_midi.note_number_to_drum_name(81) == 'Open Triangle'
    assert pretty_midi.note_number_to_drum_name(34) == ''
    assert pretty_midi.note_number_to_drum_name(82) == ''
    # Test drum_name_to_note_number
    assert pretty_midi.drum_name_to_note_number('Acoustic Bass Drum') == 35
    assert pretty_midi.drum_name_to_note_number('acousticbassdrum') == 35
    assert pretty_midi.drum_name_to_note_number('Open Triangle') == 81
    with pytest.raises(ValueError):
        pretty_midi.drum_name_to_note_number('Invalid Drum')


def test_program_name_conversions():
    # Test program_to_instrument_name
    assert pretty_midi.program_to_instrument_name(0) == 'Acoustic Grand Piano'
    assert pretty_midi.program_to_instrument_name(127) == 'Gunshot'
    with pytest.raises(ValueError):
        pretty_midi.program_to_instrument_name(-1)
    with pytest.raises(ValueError):
        pretty_midi.program_to_instrument_name(128)

    # Test instrument_name_to_program
    assert pretty_midi.instrument_name_to_program('Acoustic Grand Piano') == 0
    assert pretty_midi.instrument_name_to_program('acousticgrandpiano') == 0
    assert pretty_midi.instrument_name_to_program('Gunshot') == 127
    with pytest.raises(ValueError):
        pretty_midi.instrument_name_to_program('Invalid Instrument')

    # Test program_to_instrument_class
    assert pretty_midi.program_to_instrument_class(0) == 'Piano'
    assert pretty_midi.program_to_instrument_class(8) == 'Chromatic Percussion'
    assert pretty_midi.program_to_instrument_class(112) == 'Percussive'
    assert pretty_midi.program_to_instrument_class(127) == 'Sound Effects'
    with pytest.raises(ValueError):
        pretty_midi.program_to_instrument_class(-1)
    with pytest.raises(ValueError):
        pretty_midi.program_to_instrument_class(128)


def test_pitch_bend_conversions():
    assert pretty_midi.pitch_bend_to_semitones(0) == 0.0
    assert pretty_midi.pitch_bend_to_semitones(8192) == 2.0
    assert pretty_midi.pitch_bend_to_semitones(-8192) == -2.0
    assert pretty_midi.semitones_to_pitch_bend(0.0) == 0
    assert pretty_midi.semitones_to_pitch_bend(2.0) == 8192
    assert pretty_midi.semitones_to_pitch_bend(-2.0) == -8192
