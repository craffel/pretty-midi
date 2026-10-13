import io

import mido
import pretty_midi


def _midi_with_controllers(events):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    instrument.notes.append(pretty_midi.Note(100, 60, 2.0, 3.5))
    instrument.control_changes = [
        pretty_midi.ControlChange(number, value, time)
        for number, value, time in events]
    midi.instruments.append(instrument)
    return midi


def _written_bytes(midi):
    midi_file = io.BytesIO()
    midi.write(midi_file)
    return midi_file.getvalue()


def _written_track(midi):
    return mido.MidiFile(file=io.BytesIO(_written_bytes(midi))).tracks[1]


def _written_controllers(midi):
    return [(event.control, event.value) for event in _written_track(midi)
            if event.type == 'control_change']


def test_write_preserves_repeated_controller_order_at_same_tick():
    midi = _midi_with_controllers([(7, 100, 1.0), (7, 40, 1.0)])

    assert _written_controllers(midi) == [(7, 100), (7, 40)]


def test_write_preserves_rpn_selector_and_data_entry_order():
    controls = [(101, 0), (100, 0), (6, 2), (38, 10),
                (101, 127), (100, 127)]
    midi = _midi_with_controllers([
        (number, value, 1.0) for number, value in controls])

    assert _written_controllers(midi) == controls


def test_write_preserves_reset_before_modulation_order():
    midi = _midi_with_controllers([(121, 0, 1.0), (1, 80, 1.0)])

    assert _written_controllers(midi) == [(121, 0), (1, 80)]


def test_write_preserves_lsb_before_msb_order():
    midi = _midi_with_controllers([(33, 60, 1.0), (1, 80, 1.0)])

    assert _written_controllers(midi) == [(33, 60), (1, 80)]


def test_write_orders_quantized_controllers_by_original_time():
    midi = _midi_with_controllers([(7, 40, 1.0009), (7, 100, 1.0001)])
    assert midi.time_to_tick(1.0009) == midi.time_to_tick(1.0001)

    assert _written_controllers(midi) == [(7, 100), (7, 40)]


def test_write_orders_controllers_at_different_ticks():
    midi = _midi_with_controllers([(7, 40, 3.0), (7, 100, 1.0)])

    assert _written_controllers(midi) == [(7, 100), (7, 40)]


def test_write_keeps_controller_priority_before_note_events():
    midi = _midi_with_controllers([(7, 100, 2.0), (7, 40, 2.0)])
    events = [event.type for event in _written_track(midi)
              if event.type in ['control_change', 'note_on']]

    assert events == ['control_change', 'control_change', 'note_on',
                      'note_on']


def test_repeated_writes_are_identical_without_changing_controllers():
    midi = _midi_with_controllers([
        (7, 40, 1.0009), (101, 0, 1.0), (100, 0, 1.0),
        (7, 100, 1.0001), (6, 2, 1.0), (38, 10, 1.0),
        (121, 0, 2.0), (1, 80, 2.0)])
    instrument = midi.instruments[0]
    instrument.name = 'Controller transactions'
    instrument.pitch_bends.append(pretty_midi.PitchBend(512, 2.0))
    controllers = instrument.control_changes
    original_events = tuple(controllers)
    original_values = [
        (event.number, event.value, event.time) for event in controllers]

    first = _written_bytes(midi)
    for _ in range(3):
        assert _written_bytes(midi) == first

    assert instrument.control_changes is controllers
    assert len(controllers) == len(original_events)
    assert all(event is original for event, original in
               zip(controllers, original_events))
    assert [(event.number, event.value, event.time)
            for event in controllers] == original_values


def test_equivalent_controller_inputs_produce_identical_bytes():
    events = [(101, 0, 1.0), (100, 0, 1.0), (6, 2, 1.0),
              (38, 10, 1.0), (7, 40, 1.0009), (7, 100, 1.0001)]
    first = _midi_with_controllers(events)
    second = _midi_with_controllers(events)

    assert _written_bytes(first) == _written_bytes(second)
