"""
Utility function for converting an audio file
to a pretty_midi.PrettyMIDI object. Note that this method is nowhere close
to the state-of-the-art in automatic music transcription.
This just serves as a fun example for rough
transcription which can be expanded on for anyone motivated.
"""
from __future__ import division
import sys
import argparse
import numpy as np
import pretty_midi
import librosa


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
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


def cqt_to_piano_roll(cqt, min_midi, max_midi, threshold):
    '''Convert a CQT spectrogram into a piano roll representation by
     thresholding scaled magnitudes.

    Parameters
    ----------
    cqt : np.ndarray, shape=(max_midi-min_midi,frames), dtype=complex64
        CQT spectrogram of audio.
    min_midi : int
        Minimum MIDI note to transcribe.
    max_midi : int
        Maximum MIDI note to transcribe.
    threshold : int
        Threshold value to activate note on event, 0-127

    Returns
    -------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll representation on audio.

    '''
    piano_roll = np.abs(cqt)
    piano_roll = np.digitize(piano_roll,
                             np.linspace(piano_roll.min(),
                                         piano_roll.max(),
                                         127))
    piano_roll[piano_roll < threshold] = 0
    piano_roll = np.pad(piano_roll,
                        [(128 - max_midi, min_midi), (0, 0)],
                        'constant')
    return piano_roll


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Transcribe Audio file to MIDI file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_audio', action='store',
                        help='Path to the input Audio file')
    parser.add_argument('output_midi', action='store',
                        help='Path where the transcribed MIDI will be written')
    parser.add_argument('--program', default=0, type=int, action='store',
                        help='Program of the instrument in the output MIDI')
    parser.add_argument('--min_midi', default=24, type=int, action='store',
                        help='Minimum MIDI note to transcribe')
    parser.add_argument('--max_midi', default=107, type=int, action='store',
                        help='Maximum MIDI note to transcribe')
    parser.add_argument('--threshold', default=64, type=int, action='store',
                        help='Threshold to activate note on event, 0-127')

    parameters = vars(parser.parse_args(sys.argv[1:]))

    y, sr = librosa.load(parameters['input_audio'])
    min_midi, max_midi = parameters['min_midi'], parameters['max_midi']
    cqt = librosa.cqt(y, sr=sr, fmin=min_midi,
                      n_bins=max_midi - min_midi)
    pr = cqt_to_piano_roll(cqt, min_midi, max_midi, parameters['threshold'])
    # get audio time
    audio_time = len(y) / sr
    # get sampling frequency of cqt spectrogram
    fs = pr.shape[1]/audio_time
    pm = piano_roll_to_pretty_midi(pr, fs=fs,
                                   program=parameters['program'])
    pm.write(parameters['output_midi'])
