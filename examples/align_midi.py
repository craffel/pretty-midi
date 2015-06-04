'''
Example audio-to-MIDI alignment script.
Requires djitw https://github.com/craffel/djitw/
and librosa >= 0.4 https://github.com/bmcfee/librosa/
'''

import djitw
import librosa
import numpy as np
import sys
import argparse
import pretty_midi

# Default values for audio/CQT parameters
FS = 22050
HOP = 512
NOTE_START = 36
N_NOTES = 48


def extract_cqt(audio_data, fs, hop, note_start, n_notes):
    '''
    Compute a log-magnitude L2-normalized constant-Q-gram of some audio data.

    Parameters
    ----------
    audio_data : np.ndarray
        Audio data to compute CQT of
    fs : int
        Sampling rate of audio
    hop : int
        Hop length for CQT
    note_start : int
        Lowest MIDI note number for CQT
    n_notes : int
        Number of notes to include in the CQT

    Returns
    -------
    cqt : np.ndarray
        Log-magnitude L2-normalized CQT of the supplied audio data.
    frame_times : np.ndarray
        Times, in seconds, of each frame in the CQT
    '''
    # Compute CQT
    cqt = librosa.cqt(
        audio_data, sr=fs, hop_length=hop,
        fmin=librosa.midi_to_hz(note_start), n_bins=n_notes)
    # Transpose so that rows are spectra
    cqt = cqt.T
    # Compute log-amplitude
    cqt = librosa.logamplitude(cqt, ref_power=cqt.max())
    # L2 normalize the columns
    cqt = librosa.util.normalize(cqt, norm=2., axis=1)
    # Compute the time of each frame
    times = librosa.frames_to_time(np.arange(cqt.shape[0]), fs, hop)
    return cqt, times


def align(midi_object, audio_data, fs, hop, note_start, n_notes, penalty):
    '''
    Align a MIDI object in-place to some audio data.

    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing some MIDI content
    audio_data : np.ndarray
        Samples of some audio data
    fs : int
        audio_data's sampling rate, and the sampling rate to use when
        synthesizing MIDI
    hop : int
        Hop length for CQT
    note_start : int
        Lowest MIDI note number for CQT
    n_notes : int
        Number of notes to include in the CQT
    penalty : float
        DTW non-diagonal move penalty
    '''
    # Get synthesized MIDI audio
    midi_audio = midi_object.fluidsynth(fs=fs)
    # Compute CQ-grams for MIDI and audio
    midi_gram, midi_times = extract_cqt(
        midi_audio, fs, hop, note_start, n_notes)
    audio_gram, audio_times = extract_cqt(
        audio_data, fs, hop, note_start, n_notes)
    # Compute distance matrix; because the columns of the CQ-grams are
    # L2-normalized we can compute a cosine distance matrix via a dot product
    distance_matrix = 1 - np.dot(midi_gram, audio_gram.T)
    if penalty is None:
        penalty = distance_matrix.mean()
    # Compute lowest-cost path through distance matrix
    p, q, score = djitw.dtw(
        distance_matrix, gully=.98, penalty=penalty)
    # Adjust the timing of the MIDI object according to the alignment
    midi_object.adjust_times(midi_times[p], audio_times[q])

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Align a MIDI file to an audio file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('audio_file', action='store',
                        help='Path to the audio file to align to')
    parser.add_argument('midi_file', action='store',
                        help='Path to the MIDI file to align to the audio')
    parser.add_argument('output_file', action='store',
                        help='Path where the aligned MIDI will be written')
    parser.add_argument('--output_audio', default=None, type=str,
                        action='store',
                        help='Path where an audio file consisting of the '
                             'synthesized aligned MIDI in one channel and the '
                             'audio data in the other will be written')
    parser.add_argument('--fs', default=FS, type=int, action='store',
                        help='Global audio sampling rate to use')
    parser.add_argument('--hop', default=HOP, type=int, action='store',
                        help='CQT hop length')
    parser.add_argument('--note_start', default=NOTE_START, type=int,
                        action='store', help='Lowest CQT MIDI note')
    parser.add_argument('--n_notes', default=N_NOTES, type=int, action='store',
                        help='Number of notes in each CQT')
    parser.add_argument('--penalty', default=None, type=float,
                        action='store', help='DTW non-diagonal move penalty.  '
                        'By default, uses the mean of the distance matrix.')

    parameters = vars(parser.parse_args(sys.argv[1:]))
    print "Loading {} ...".format(parameters['audio_file'])
    audio_data, _ = librosa.load(parameters['audio_file'], sr=parameters['fs'])
    print "Loading {} ...".format(parameters['midi_file'])
    midi_object = pretty_midi.PrettyMIDI(parameters['midi_file'])
    print "Aligning {} to {} ...".format(parameters['audio_file'],
                                         parameters['midi_file'])
    align(midi_object, audio_data, parameters['fs'], parameters['hop'],
          parameters['note_start'], parameters['n_notes'],
          parameters['penalty'])
    print "Writing {} ...".format(parameters['output_file'])
    midi_object.write(parameters['output_file'])
    if parameters['output_audio']:
        print "Writing {} ...".format(parameters['output_audio'])
        # Re-synthesize the aligned mIDI
        midi_audio_aligned = midi_object.fluidsynth(fs=parameters['fs'])
        # Adjust to the same size as audio
        if midi_audio_aligned.shape[0] > audio_data.shape[0]:
            midi_audio_aligned = midi_audio_aligned[:audio_data.shape[0]]
        else:
            trim_amount = audio_data.shape[0] - midi_audio_aligned.shape[0]
            midi_audio_aligned = np.append(midi_audio_aligned,
                                           np.zeros(trim_amount))
        # Write out a .wav with aligned MIDI/audio in each channel
        librosa.output.write_wav(parameters['output_audio'],
                                 np.vstack([midi_audio_aligned, audio_data]),
                                 parameters['fs'])
