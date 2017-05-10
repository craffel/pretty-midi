from __future__ import division
"""
Utility function for converting a piano roll (integer matrix of shape
(n_notes, n_frames)) to a pretty_midi.PrettyMIDI object. Note that this is a
lossy process because certain information in a MIDI file (such as different
instruments, pitch bends, overlapping notes, control changes, etc) cannot
be stored in the piano roll matrix. To demonstrate the lossiness, this script
includes a demonstration of parsing a MIDI file, constructing a piano roll
matrix, and then creating a new MIDI from the piano roll.
"""
import pretty_midi
import numpy as np
import sys
import argparse


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=1):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,time), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                    velocity=prev_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Translate MIDI file to piano roll and back',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_midi', action='store',
                        help='Path to the input MIDI file')
    parser.add_argument('output_midi', action='store',
                        help='Path where the translated MIDI will be written')
    parser.add_argument('--fs', default=100, type=int, action='store',
                        help='Sampling rate to use between conversions')
    parser.add_argument('--program', default=0, type=int, action='store',
                        help='Program of the instrument')

    parameters = vars(parser.parse_args(sys.argv[1:]))
    pm = pretty_midi.PrettyMIDI(parameters['input_midi'])
    pr = pm.get_piano_roll(fs=parameters['fs'])
    new_pm = piano_roll_to_pretty_midi(pr, fs=parameters['fs'],
                                       program=parameters['program'])
    new_pm.write(parameters['output_midi'])
