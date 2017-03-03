"""Synthesize a MIDI file, chiptunes-style! (using pure numpy and scipy)

Includes functions for synthesizing different drum types, bass instruments, and
all other instruments.  Also for auto-arpeggiating chords.
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile
import pretty_midi
import argparse
import sys


def tonal(fs, length, frequency, nonlinearity=1.):
    '''
    Synthesize a tonal drum.

    Parameters
    ----------
    fs : int
        Sampling frequency
    length : int
        Length, in samples, of drum sound
    frequency : float
        Frequency, in Hz, of the drum
    nonlinearity : float
        Gain to apply for nonlinearity, default 1.

    Returns
    -------
    drum_data : np.ndarray
        Synthesized drum data
    '''
    # Amplitude envelope, decaying exponential
    amp_envelope = np.exp(np.linspace(0, -10, length))
    # Pitch envelope, starting with linear decay
    pitch_envelope = np.linspace(1.0, .99, length)
    # Also a quick exponential drop at the beginning for a click
    pitch_envelope *= 100*np.exp(np.linspace(0, -100*frequency, length)) + 1
    # Generate tone
    drum_data = amp_envelope*np.sin(
        2*np.pi*frequency*pitch_envelope*np.arange(length)/float(fs))
    # Filter with leaky integrator with 3db point ~= note frequency
    alpha = 1 - np.exp(-2*np.pi*(frequency)/float(fs))
    drum_data = scipy.signal.lfilter([alpha], [1, alpha - 1], drum_data)
    # Apply nonlinearity
    drum_data = np.tanh(nonlinearity*drum_data)
    return drum_data


def noise(length):
    '''
    Synthesize a noise drum.

    Parameters
    ----------
    length : int
        Number of samples to synthesize.

    Returns
    -------
    drum_data : np.ndarray
        Synthesized drum data
    '''
    # Amplitude envelope, decaying exponential
    amp_envelope = np.exp(np.linspace(0, -10, length))
    # Synthesize gaussian random noise
    drum_data = amp_envelope*np.random.randn(length)
    return drum_data


def synthesize_drum_instrument(instrument, fs=44100):
    '''
    Synthesize a pretty_midi.Instrument object with drum sounds.

    Parameters
    ----------
    instrument : pretty_midi.Instrument
        Instrument to synthesize

    Returns
    -------
    synthesized : np.ndarray
        Audio data of the instrument synthesized
    '''
    # Allocate audio data
    synthesized = np.zeros(int((instrument.get_end_time() + 1)*fs))
    for note in instrument.notes:
        # Get the name of the drum
        drum_name = pretty_midi.note_number_to_drum_name(note.pitch)
        # Based on the drum name, synthesize using the tonal or noise functions
        if drum_name in ['Acoustic Bass Drum', 'Bass Drum 1']:
            d = tonal(fs, fs/2, 80, 8.)
        elif drum_name in ['Side Stick']:
            d = tonal(fs, fs/20, 400, 8.)
        elif drum_name in ['Acoustic Snare', 'Electric Snare']:
            d = .4*tonal(fs, fs/10, 200, 20.) + .6*noise(fs/10)
        elif drum_name in ['Hand Clap', 'Vibraslap']:
            d = .1*tonal(fs, fs/10, 400, 8.) + .9*noise(fs/10)
        elif drum_name in ['Low Floor Tom', 'Low Tom', 'Low Bongo',
                           'Low Conga', 'Low Timbale']:
            d = tonal(fs, fs/4, 120, 8.)
        elif drum_name in ['Closed Hi Hat', 'Cabasa', 'Maracas',
                           'Short Guiro']:
            d = noise(fs/20)
        elif drum_name in ['High Floor Tom', 'High Tom', 'Hi Bongo',
                           'Open Hi Conga', 'High Timbale']:
            d = tonal(fs, fs/4, 480, 4.)
        elif drum_name in ['Pedal Hi Hat', 'Open Hi Hat', 'Crash Cymbal 1',
                           'Ride Cymbal 1', 'Chinese Cymbal', 'Crash Cymbal 2',
                           'Ride Cymbal 2', 'Tambourine',  'Long Guiro',
                           'Splash Cymbal']:
            d = .8*noise(fs)
        elif drum_name in ['Low-Mid Tom']:
            d = tonal(fs, fs/4, 240, 4.)
        elif drum_name in ['Hi-Mid Tom']:
            d = tonal(fs, fs/4, 360, 4.)
        elif drum_name in ['Mute Hi Conga', 'Mute Cuica', 'Cowbell',
                           'Low Agogo', 'Low Wood Block']:
            d = tonal(fs, fs/10, 480, 4.)
        elif drum_name in ['Ride Bell', 'High Agogo', 'Claves',
                           'Hi Wood Block']:
            d = tonal(fs, fs/20, 960, 4.)
        elif drum_name in ['Short Whistle']:
            d = tonal(fs, fs/4, 480, 1.)
        elif drum_name in ['Long Whistle']:
            d = tonal(fs, fs, 480, 1.)
        elif drum_name in ['Mute Triangle']:
            d = tonal(fs, fs/10, 1960, 1.)
        elif drum_name in ['Open Triangle']:
            d = tonal(fs, fs, 1960, 1.)
        else:
            if drum_name is not '':
                # This should never happen
                print 'Unexpected drum {}'.format(drum_name)
            continue
        # Add in the synthesized waveform
        start = int(note.start*fs)
        synthesized[start:start+d.size] += d*note.velocity
    return synthesized


def arpeggiate_instrument(instrument, arpeggio_time):
    '''
    Arpeggiate the notes of an instrument.

    Parameters
    ----------
    inst : pretty_midi.Instrument
        Instrument object.
    arpeggio_time : float
        Time, in seconds, of each note in the arpeggio

    Returns
    -------
    inst_arpeggiated : pretty_midi.Instrument
        Instrument with the notes arpeggiated.
    '''
    # Make a copy of the instrument
    inst_arpeggiated = pretty_midi.Instrument(program=instrument.program,
                                              is_drum=instrument.is_drum)
    for bend in instrument.pitch_bends:
        inst_arpeggiated.pitch_bends.append(bend)
    n = 0
    while n < len(instrument.notes):
        # Collect notes which are in this chord
        chord_notes = [(instrument.notes[n].pitch,
                        instrument.notes[n].velocity)]
        m = n + 1
        while m < len(instrument.notes):
            # It's in the chord if it starts before the current note ends
            if instrument.notes[m].start < instrument.notes[n].end:
                # Add in the pitch and velocity
                chord_notes.append((instrument.notes[m].pitch,
                                    instrument.notes[m].velocity))
                # Move the start time of the note up so it gets used next time
                if instrument.notes[m].end > instrument.notes[n].end:
                    instrument.notes[m].start = instrument.notes[n].end
            m += 1
        # Arpeggiate the collected notes
        time = instrument.notes[n].start
        pitch_index = 0
        if len(chord_notes) > 2:
            while time < instrument.notes[n].end:
                # Get the pitch and velocity of this note, but mod the index
                # to circulate
                pitch, velocity = chord_notes[pitch_index % len(chord_notes)]
                # Add this note to the new instrument
                inst_arpeggiated.notes.append(
                    pretty_midi.Note(velocity, pitch, time,
                                     time + arpeggio_time))
                # Next pitch next time
                pitch_index += 1
                # Move forward by the supplied amount
                time += arpeggio_time
        else:
            inst_arpeggiated.notes.append(instrument.notes[n])
            time = instrument.notes[n].end
        n += 1
        # Find the next chord
        while (n < len(instrument.notes) and
               instrument.notes[n].start + arpeggio_time <= time):
            n += 1
    return inst_arpeggiated


def chiptunes_synthesize(midi, fs=44100):
    '''
    Synthesize a pretty_midi.PrettyMIDI object chiptunes style.

    Parameters
    ----------
    midi : pretty_midi.PrettyMIDI
        PrettyMIDI object to synthesize
    fs : int
        Sampling rate of the synthesized audio signal, default 44100

    Returns
    -------
    synthesized : np.ndarray
        Waveform of the MIDI data, synthesized at fs
    '''
    # If there are no instruments, return an empty array
    if len(midi.instruments) == 0:
        return np.array([])
    # Get synthesized waveform for each instrument
    waveforms = []
    for inst in midi.instruments:
        # Synthesize as drum
        if inst.is_drum:
            waveforms.append(synthesize_drum_instrument(inst, fs=fs))
        else:
            # Call it a bass instrument when no notes are over 48 (130hz)
            # or the program's name has the word "bass" in it
            is_bass = (
                np.max([n.pitch for i in midi.instruments
                        for n in i.notes]) < 48 or
                'Bass' in pretty_midi.program_to_instrument_name(inst.program))
            if is_bass:
                # Synthesize as a sine wave (should be triangle!)
                audio = inst.synthesize(fs=fs, wave=np.sin)
                # Quantize to 5-bit
                audio = np.digitize(
                    audio, np.linspace(-audio.min(), audio.max(), 32))
                waveforms.append(audio)
            else:
                # Otherwise, it's a harmony/lead instrument, so arpeggiate it
                # Arpeggio time of 30ms seems to work well
                inst_arpeggiated = arpeggiate_instrument(inst, .03)
                # These instruments sound louder because they're square,
                # so scale down
                waveforms.append(.5*inst_arpeggiated.synthesize(
                    fs=fs, wave=scipy.signal.square))
    # Allocate output waveform, with #sample = max length of all waveforms
    synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
    # Sum all waveforms in
    for waveform in waveforms:
        synthesized[:waveform.shape[0]] += waveform
    # Normalize
    synthesized /= np.abs(synthesized).max()
    return synthesized


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Synthesize a MIDI file, chiptunes style.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('midi_file', action='store',
                        help='Path to the MIDI file to synthesize')
    parser.add_argument('output_file', action='store',
                        help='Path where the synthesized wav will be written')
    parser.add_argument('--fs', default=44100, type=int, action='store',
                        help='Output sampling rate to use')

    # Parse command line arguments
    parameters = vars(parser.parse_args(sys.argv[1:]))
    print "Synthesizing {} ...".format(parameters['midi_file'])
    # Load in MIDI data and synthesize using chiptunes_synthesize
    midi_object = pretty_midi.PrettyMIDI(parameters['midi_file'])
    synthesized = chiptunes_synthesize(midi_object, parameters['fs'])
    print "Writing {} ...".format(parameters['output_file'])
    # Write out
    scipy.io.wavfile.write(
        parameters['output_file'], parameters['fs'], synthesized)
