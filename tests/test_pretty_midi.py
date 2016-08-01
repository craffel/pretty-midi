import pretty_midi
import numpy as np


def test_get_beats():
    pm = pretty_midi.PrettyMIDI()
    # Add a note to force get_end_time() to be non-zero
    i = pretty_midi.Instrument(0)
    i.notes.append(pretty_midi.Note(100, 100, 0.3, 10.4))
    pm.instruments.append(i)
    # pretty_midi assumes 120 bpm unless otherwise specified
    assert np.allclose(pm.get_beats(),
                       np.arange(0, pm.get_end_time(), 60./120.))
    # Testing starting from a different beat time
    assert np.allclose(pm.get_beats(.2),
                       np.arange(0, pm.get_end_time(), 60./120.) + .2)
    # Testing a tempo change
    change_bpm = 93.
    change_time = 4.4
    pm._tick_scales.append(
        (pm.time_to_tick(change_time), 60./(change_bpm*pm.resolution)))
    pm._update_tick_to_time(pm.time_to_tick(pm.get_end_time()))
    # Track at 120 bpm up to the tempo change time
    expected_beats = np.arange(0, change_time, 60./120.)
    # BPM switches (4.5 - 4.4)/(60./120.) of the way through
    expected_beats = np.append(
        expected_beats,
        change_time + (4.5 - change_time)/(60./120.)*60./change_bpm)
    # From there, use the new bpm
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 60./change_bpm,
                                         pm.get_end_time(), 60./change_bpm))
    assert np.allclose(pm.get_beats(), expected_beats)
    # When requesting a start_time after the tempo change, make sure we just
    # track as normal
    assert np.allclose(
        pm.get_beats(change_time + .1),
        np.arange(change_time + .1, pm.get_end_time(), 60./change_bpm))
    # Add a time signature change, which forces beat tracking to restart
    pm.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 2.1))
    # Track at 120 bpm up to time signature change
    expected_beats = np.arange(0, 2.1, 60./120.)
    # Now track, restarting from time signature change time
    expected_beats = np.append(expected_beats,
                               np.arange(2.1, change_time, 60./120.))
    # BPM switches (4.6 - 4.4)/(60./120.) of the way through
    expected_beats = np.append(
        expected_beats,
        change_time + (4.6 - change_time)/(60./120.)*60./change_bpm)
    # From there, use the new bpm
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 60./change_bpm,
                                         pm.get_end_time(), 60./change_bpm))
    assert np.allclose(pm.get_beats(), expected_beats)
    # When there are two time signature changes, make sure both get included
    pm.time_signature_changes.append(pretty_midi.TimeSignature(5, 4, 1.9))
    expected_beats[expected_beats == 2.] = 1.9
    assert np.allclose(pm.get_beats(), expected_beats)
    # Request a start time after time time signature change
    expected_beats = np.arange(2.2, change_time, 60./120.)
    expected_beats = np.append(
        expected_beats,
        change_time + (4.7 - change_time)/(60./120.)*60./change_bpm)
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 60./change_bpm,
                                         pm.get_end_time(), 60./change_bpm))
    assert np.allclose(pm.get_beats(2.2), expected_beats)


def test_get_downbeats():
    pm = pretty_midi.PrettyMIDI()
    # Add a note to force get_end_time() to be non-zero
    i = pretty_midi.Instrument(0)
    i.notes.append(pretty_midi.Note(100, 100, 0.3, 20.4))
    pm.instruments.append(i)
    # pretty_midi assumes 120 bpm, 4/4 unless otherwise specified
    assert np.allclose(pm.get_downbeats(),
                       np.arange(0, pm.get_end_time(), 4*60./120.))
    # Testing starting from a different beat time
    assert np.allclose(pm.get_downbeats(.2),
                       np.arange(0, pm.get_end_time(), 4*60./120.) + .2)
    # Testing a tempo change
    change_bpm = 93.
    change_time = 8.4
    pm._tick_scales.append(
        (pm.time_to_tick(change_time), 60./(change_bpm*pm.resolution)))
    pm._update_tick_to_time(pm.time_to_tick(pm.get_end_time()))
    # Track at 120 bpm up to the tempo change time
    expected_beats = np.arange(0, change_time, 4*60./120.)
    # BPM switches (4.5 - 4.4)/(60./120.) of the way through
    expected_beats = np.append(
        expected_beats,
        change_time + (10. - change_time)/(4*60./120.)*4*60./change_bpm)
    # From there, use the new bpm
    expected_beats = np.append(
        expected_beats, np.arange(expected_beats[-1] + 4*60./change_bpm,
                                  pm.get_end_time(), 4*60./change_bpm))
    assert np.allclose(pm.get_downbeats(), expected_beats)
    # When requesting a start_time after the tempo change, make sure we just
    # track as normal
    assert np.allclose(
        pm.get_downbeats(change_time + .1),
        np.arange(change_time + .1, pm.get_end_time(), 4*60./change_bpm))
    # Add a time signature change, which forces beat tracking to restart
    pm.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 2.1))
    # Track at 120 bpm up to time signature change
    expected_beats = np.arange(0, 2.1, 4*60./120.)
    # Now track, restarting from time signature change time
    expected_beats = np.append(expected_beats,
                               np.arange(2.1, change_time, 3*60./120.))
    # BPM switches (4.6 - 4.4)/(60./120.) of the way through
    expected_beats = np.append(
        expected_beats,
        change_time + (9.6 - change_time)/(3*60./120.)*3*60./change_bpm)
    # From there, use the new bpm
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 3*60./change_bpm,
                                         pm.get_end_time(), 3*60./change_bpm))
    assert np.allclose(pm.get_downbeats(), expected_beats)
    # When there are two time signature changes, make sure both get included
    pm.time_signature_changes.append(pretty_midi.TimeSignature(5, 4, 1.9))
    expected_beats[expected_beats == 2.] = 1.9
    assert np.allclose(pm.get_downbeats(), expected_beats)
    # Request a start time after time time signature change
    expected_beats = np.arange(2.2, change_time, 3*60./120.)
    expected_beats = np.append(
        expected_beats,
        change_time + (9.7 - change_time)/(3*60./120.)*3*60./change_bpm)
    expected_beats = np.append(expected_beats,
                               np.arange(expected_beats[-1] + 3*60./change_bpm,
                                         pm.get_end_time(), 3*60./change_bpm))
    assert np.allclose(pm.get_downbeats(2.2), expected_beats)


def test_adjust_times():
    # Simple tests for adjusting note times
    def simple():
        pm = pretty_midi.PrettyMIDI()
        i = pretty_midi.Instrument(0)
        # Create 9 notes, at times [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for n, start in enumerate(range(1, 10)):
            i.notes.append(pretty_midi.Note(100, 100 + n, start, start + .5))
        pm.instruments.append(i)
        return pm
    # Test notes are interpolated as expected
    pm = simple()
    pm.adjust_times([0, 10], [5, 20])
    for note, start in zip(pm.instruments[0].notes, 1.5*np.arange(1, 10) + 5):
        assert note.start == start
    # Test notes are all ommitted when adjustment range doesn't cover them
    pm = simple()
    pm.adjust_times([10, 20], [5, 10])
    assert len(pm.instruments[0].notes) == 0
    # Test repeated mapping times
    pm = simple()
    pm.adjust_times([0, 5, 6.5, 10], [5, 10, 10, 17])
    # Original times  [1, 2, 3, 4,  7,  8,  9]
    # The notes at times 5 and 6 have their durations squashed to zero
    expected_starts = [6, 7, 8, 9, 11, 13, 15]
    assert np.allclose(
        [n.start for n in pm.instruments[0].notes], expected_starts)
    pm = simple()
    pm.adjust_times([0, 5, 5, 10], [5, 10, 12, 17])
    # Original times  [1, 2, 3, 4,  5,  6,  7,  8,  9]
    expected_starts = [6, 7, 8, 9, 12, 13, 14, 15, 16]
    assert np.allclose(
        [n.start for n in pm.instruments[0].notes], expected_starts)

    # Complicated example
    pm = simple()
    # Include pitch bends and control changes to test adjust_events
    pm.instruments[0].pitch_bends.append(pretty_midi.PitchBend(100, 1.))
    # Include event which fall within omitted region
    pm.instruments[0].pitch_bends.append(pretty_midi.PitchBend(200, 7.))
    pm.instruments[0].pitch_bends.append(pretty_midi.PitchBend(0, 7.1))
    # Include event which falls outside of the track
    pm.instruments[0].pitch_bends.append(pretty_midi.PitchBend(10, 10.))
    pm.instruments[0].control_changes.append(
        pretty_midi.ControlChange(0, 0, .5))
    pm.instruments[0].control_changes.append(
        pretty_midi.ControlChange(0, 1, 5.5))
    pm.instruments[0].control_changes.append(
        pretty_midi.ControlChange(0, 2, 7.5))
    pm.instruments[0].control_changes.append(
        pretty_midi.ControlChange(0, 3, 20.))
    # Include track-level meta events to test adjust_meta
    pm.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, .1))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 5.2))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(6, 4, 6.2))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(5, 4, 15.3))
    pm.key_signature_changes.append(pretty_midi.KeySignature(1, 1.))
    pm.key_signature_changes.append(pretty_midi.KeySignature(2, 6.2))
    pm.key_signature_changes.append(pretty_midi.KeySignature(3, 7.2))
    pm.key_signature_changes.append(pretty_midi.KeySignature(4, 12.3))
    # Add in tempo changes - 100 bpm at 0s
    pm._tick_scales[0] = (0, 60./(100*pm.resolution))
    # 110 bpm at 6s
    pm._tick_scales.append((2200, 60./(110*pm.resolution)))
    # 120 bpm at 8.1s
    pm._tick_scales.append((3047, 60./(120*pm.resolution)))
    # 150 bpm at 8.3s
    pm._tick_scales.append((3135, 60./(150*pm.resolution)))
    # 80 bpm at 9.3s
    pm._tick_scales.append((3685, 60./(80*pm.resolution)))
    pm._update_tick_to_time(20000)

    # Adjust times, with a collapsing section in original and new times
    pm.adjust_times([2., 3.1, 3.1, 5.1, 7.5, 10],
                    [5., 6., 7., 8.5, 8.5, 11])

    # Original tempo change times: [0, 6, 8.1, 8.3, 9.3]
    # Plus tempo changes at each of new_times which are not collapsed
    # Plus tempo change at 0s by default
    expected_times = [0., 5., 6., 8.5,
                      8.5 + (6 - 5.1)*(11 - 8.5)/(10 - 5.1),
                      8.5 + (8.1 - 5.1)*(11 - 8.5)/(10 - 5.1),
                      8.5 + (8.3 - 5.1)*(11 - 8.5)/(10 - 5.1),
                      8.5 + (9.3 - 5.1)*(11 - 8.5)/(10 - 5.1)]
    # Tempos scaled by differences in timing, plus 120 bpm at the beginning
    expected_tempi = [120., 100*(3.1 - 2)/(6 - 5),
                      100*(5.1 - 3.1)/(8.5 - 6),
                      100*(10 - 5.1)/(11 - 8.5),
                      110*(10 - 5.1)/(11 - 8.5),
                      120*(10 - 5.1)/(11 - 8.5),
                      150*(10 - 5.1)/(11 - 8.5),
                      80*(10 - 5.1)/(11 - 8.5)]
    change_times, tempi = pm.get_tempo_changes()
    # Due to the fact that tempo change times must occur at discrete ticks, we
    # must raise the relative tolerance when comparing
    assert np.allclose(expected_times, change_times, rtol=.001)
    assert np.allclose(expected_tempi, tempi, rtol=.002)

    # Test that all other events were interpolated as expected
    note_starts = [5., 5 + 1/1.1, 7 + .9/(2/1.5), 7 + 1.9/(2/1.5), 8.5 + .5,
                   8.5 + 1.5]
    note_ends = [5 + .5/1.1, 7 + .4/(2/1.5), 7 + 1.4/(2/1.5), 8.5, 9 + .5,
                 10 + .5]
    note_pitches = [101, 102, 103, 104, 107, 108, 109]
    for note, s, e, p in zip(pm.instruments[0].notes, note_starts, note_ends,
                             note_pitches):
        assert note.start == s
        assert note.end == e
        assert note.pitch == p

    bend_times = [5., 8.5, 8.5]
    bend_pitches = [100, 200, 0]
    for bend, t, p in zip(pm.instruments[0].pitch_bends, bend_times,
                          bend_pitches):
        assert bend.time == t
        assert bend.pitch == p

    cc_times = [5., 8.5, 8.5]
    cc_values = [0, 1, 2]
    for cc, t, v in zip(pm.instruments[0].control_changes, cc_times,
                        cc_values):
        assert cc.time == t
        assert cc.value == v

    # The first time signature change will be placed at the first interpolated
    # downbeat location - so, start by computing the location of the first
    # downbeat after the start of original_times, then interpolate it
    first_downbeat_after = .1 + 2*3*60./100.
    first_ts_time = 7 + (first_downbeat_after - 3.1)/(2/1.5)
    ts_times = [first_ts_time, 8.5, 8.5]
    ts_numerators = [3, 4, 6]
    for ts, t, n in zip(pm.time_signature_changes, ts_times, ts_numerators):
        assert ts.time == t
        assert ts.numerator == n

    ks_times = [5., 8.5, 8.5]
    ks_keys = [1, 2, 3]
    for ks, t, k in zip(pm.key_signature_changes, ks_times, ks_keys):
        assert ks.time == t
        assert ks.key_number == k
