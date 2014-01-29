# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Utility functions for handling MIDI data in an easy to read/manipulate format
'''

# <codecell>

import midi
import numpy as np
try:
    import fluidsynth
    _HAS_FLUIDSYNTH = True
except ImportError:
    _HAS_FLUIDSYNTH = False
import os
import itertools
import warnings

# <codecell>

class PrettyMIDI(object):
    '''
    A container for MIDI data in a nice format.
    
    Members:
        instruments - list of pretty_midi.Instrument objects, corresponding to the instruments which play in the MIDI file
    '''
    def __init__(self, midi_data):
        '''
        Initialize the PrettyMIDI container with some midi data
        
        Input:
            midi_data - midi.FileReader object
        '''
        # Convert tick values in midi_data to absolute, a useful thing.
        midi_data.make_ticks_abs()
        
        # Store the resolution for later use
        self.resolution = midi_data.resolution
        
        # Populate the list of tempo changes (tick scales)
        self._get_tempo_changes(midi_data)
        # Update the array which maps ticks to time
        max_tick = max([max([event.tick for event in track]) for track in midi_data]) + 1
        self._update_tick_to_time(max_tick)
        # Check that there are no tempo change events on any tracks other than track 0
        if sum([sum([event.name == 'Set Tempo' for event in track]) for track in midi_data[1:]]):
            warnings.warn("Tempo change events found on non-zero tracks.  \
This is not a valid type 0 or type 1 MIDI file.  Timing may be wrong.",
                          RuntimeWarning)
            
        # Populate the list of instruments
        self._get_instruments(midi_data)
        
    def _get_tempo_changes(self, midi_data):
        '''
        Populates self.tick_scales with tuples of (tick, tick_scale)
        
        Input:
            midi_data - midi.FileReader object
        '''
        # MIDI data is given in "ticks".  We need to convert this to clock seconds.
        # The conversion factor has to do with the BPM, which may change over time.
        # So, create a list of tuples, (time, tempo) which denotes the tempo over time
        # By default, set the tempo to 120 bpm, starting at time 0
        self.tick_scales = [(0, 60.0/(120.0*midi_data.resolution))]
        # Keep track of the absolute tick value of the previous tempo change event
        lastTick = 0
        # For SMF file type 0, all events are on track 0.
        # For type 1, all tempo events should be on track 1.
        # Everyone ignores type 2.
        # So, just look at events on track 0
        for event in midi_data[0]:
            if event.name == 'Set Tempo':
                # Only allow one tempo change event at the beginning
                if event.tick == 0:
                    self.tick_scales = [(0, 60.0/(event.get_bpm()*midi_data.resolution))]
                else:
                    # Get time and BPM up to this point
                    last_tick, last_tick_scale = self.tick_scales[-1]
                    tick_scale = 60.0/(event.get_bpm()*midi_data.resolution)
                    # Ignore repetition of BPM, which happens often
                    if tick_scale != last_tick_scale:
                        self.tick_scales.append( (event.tick, tick_scale) )
    
    def _update_tick_to_time(self, max_tick):
        '''
        Creates tick_to_time, an array which maps ticks to time, from tick 0 to max_tick
        
        Input:
            max_tick - last tick to compute time for
        '''
        # Allocate tick to time array - indexed by tick from 0 to max_tick
        self.tick_to_time = np.zeros( max_tick )
        # Keep track of the end time of the last tick in the previous interval
        last_end_time = 0
        # Cycle through intervals of different tempii
        for (start_tick, tick_scale), (end_tick, _) in zip(self.tick_scales[:-1], self.tick_scales[1:]):
            # Convert ticks in this interval to times
            self.tick_to_time[start_tick:end_tick + 1] = last_end_time + tick_scale*np.arange(end_tick - start_tick + 1)
            # Update the time of the last tick in this interval
            last_end_time = self.tick_to_time[end_tick]
        # For the final interval, use the final tempo setting and ticks from the final tempo setting until max_tick
        start_tick, tick_scale = self.tick_scales[-1]
        self.tick_to_time[start_tick:] = last_end_time + tick_scale*np.arange(max_tick - start_tick)
        
    def _get_instruments(self, midi_data):
        '''
        Populates the list of instruments in midi_data.
        
        Input:
            midi_data - midi.FileReader object
        '''
        # Initialize empty list of instruments
        self.instruments = []
        for track in midi_data:
            # Keep track of last note on location: key = (instrument, is_drum, note), value = (note on time, velocity)
            last_note_on = {}
            # Keep track of pitch bends: key = (instrument, is_drum) value = (pitch bend time, pitch bend amount)
            pitch_bends = {}
            # Keep track of which instrument is playing in each channel - initialize to program 0 for all channels
            current_instrument = np.zeros( 16, dtype=np.int )
            for event in track:
                # Look for program change events
                if event.name == 'Program Change':
                    # Update the instrument for this channel
                    current_instrument[event.channel] = event.data[0]
                # Note ons are note on events with velocity > 0
                elif event.name == 'Note On' and event.velocity > 0:
                    # Check whether this event is for the drum channel
                    is_drum = (event.channel == 9)
                    # Store this as the last note-on location
                    note_on_index = (current_instrument[event.channel], is_drum, event.pitch)
                    last_note_on[note_on_index] = (self.tick_to_time[event.tick], event.velocity)
                # Note offs can also be note on events with 0 velocity
                elif event.name == 'Note Off' or (event.name == 'Note On' and event.velocity == 0):
                    # Check whether this event is for the drum channel
                    is_drum = (event.channel == 9)
                    # Check that a note-on exists (ignore spurious note-offs)
                    if (current_instrument[event.channel], is_drum, event.pitch) in last_note_on:
                        # Get the start/stop times and velocity of this note
                        start, velocity = last_note_on[(current_instrument[event.channel], is_drum, event.pitch)]
                        end = self.tick_to_time[event.tick]
                        # Check that the instrument exists
                        instrument_exists = False
                        for instrument in self.instruments:
                            # Find the right instrument
                            if instrument.program == current_instrument[event.channel] and instrument.is_drum == is_drum:
                                instrument_exists = True
                                # Add this note event
                                instrument.events.append(Note(velocity, event.pitch, start, end))
                        # Create the instrument if none was found
                        if not instrument_exists:
                            # Create a new instrument
                            self.instruments.append(Instrument(current_instrument[event.channel], is_drum))
                            instrument = self.instruments[-1]
                            # Add the note to the new instrument
                            instrument.events.append(Note(event.velocity, event.pitch, start, end))
                        # Remove the last note on for this instrument
                        del last_note_on[(current_instrument[event.channel], is_drum, event.pitch)]
                # Store pitch bends
                elif event.name == 'Pitch Wheel':
                    # Check whether this event is for the drum channel
                    is_drum = (event.channel == 9)
                    # Convert to relative pitch in semitones
                    pitch_bend = 2*event.pitch/8192.0
                    for instrument in self.instruments:
                        # Find the right instrument
                        if instrument.program == current_instrument[event.channel] and instrument.is_drum == is_drum:
                            # Store pitch bend information
                            instrument.pitch_changes.append((self.tick_to_time[event.tick], pitch_bend))
        
    def get_tempo_changes(self):
        '''
        Return arrays of tempo changes and their times.  This is direct from the MIDI file.
        
        Output:
            tempo_change_times - Times, in seconds, where the tempo changes.
            tempii - np.ndarray of tempos, same size as tempo_change_times
        '''
        # Pre-allocate return arrays
        tempo_change_times = np.zeros(len(self.tick_scales))
        tempii = np.zeros(len(self.tick_scales))
        for n, (tick, tick_scale) in enumerate(self.tick_scales):
            # Convert tick of this tempo change to time in seconds
            tempo_change_times[n] = self.tick_to_time[tick]
            # Convert tick scale to a tempo
            tempii[n] = 60.0/(tick_scale*self.resolution)
        return tempo_change_times, tempii

    def get_end_time(self):
        '''
        Returns the time of the end of this MIDI file (latest note-off event).
        
        Output:
            end_time - Time, in seconds, where this MIDI file ends
        '''
        # Cycle through all notes from all instruments and find the largest
        end_time = 0.0
        for instrument in self.instruments:
            for note in instrument.events:
                if note.end > end_time:
                    end_time = note.end
        return end_time
        
    def estimate_tempii(self):
        '''
        Return an empirical estimate of tempos in the piece
        Based on "Automatic Extraction of Tempo and Beat from Expressive Performance", Dixon 2001
        
        Output:
            tempo - Estimated tempo, in bpm
        '''
        # Grab the list of onsets
        onsets = self.get_onsets()
        # Compute inner-onset intervals
        ioi = np.diff( onsets )
        # "Rhythmic information is provided by IOIs in the range of approximately 50ms to 2s (Handel, 1989)"
        ioi = ioi[ioi > .05]
        ioi = ioi[ioi < 2]
        # Normalize all iois into the range 30...300bpm
        for n in xrange(ioi.shape[0]):
            while ioi[n] < .2:
                ioi[n] *= 2
        # Array of inner onset interval cluster means
        clusters = np.array([])
        # Number of iois in each cluster
        cluster_counts = np.array([])
        for interval in ioi:
            # If this ioi falls within a cluster (threshold is 25ms)
            if (np.abs(clusters - interval) < .025).any():
                k = np.argmin(clusters - interval)
                # Update cluster mean
                clusters[k] = (cluster_counts[k]*clusters[k] + interval)/(cluster_counts[k] + 1)
                # Update number of elements in cluster
                cluster_counts[k] += 1
            # No cluster is close, make a new one
            else:
                clusters = np.append(clusters, interval)
                cluster_counts = np.append(cluster_counts, 1.)
        # Sort the cluster list by 
        cluster_sort = np.argsort(cluster_counts)[::-1]
        clusters = clusters[cluster_sort]
        cluster_counts = cluster_counts[cluster_sort]
        # Normalize the cluster scores
        cluster_counts /= cluster_counts.max()
        return 60./clusters, cluster_counts
    
    def estimate_tempo(self):
        '''
        Returns the best tempo estimate from estimate_tempii(), for convenience
        
        Output:
            tempo - Estimated tempo, in bpm
        '''
        return self.estimate_tempii()[0][0]
    
    def get_beats(self):
        '''
        Return a list of beat locations, according to the MIDI file tempo changes.
        Will not be correct if the MIDI data has been modified without changing tempo information.
        
        Output:
            beats - np.ndarray of beat locations, in seconds
        '''
        # Get a sorted list of all notes from all instruments
        note_list = [note for instrument in self.instruments for note in instrument.events]
        note_list.sort(key=lambda note: note.start)
        # Get tempo changs and tempos
        tempo_change_times, tempii = self.get_tempo_changes()
        
        def beat_track_using_tempo(start_time):
            ''' Starting from start_time, place beats according to the MIDI file's designated tempo changes '''
            # Create beat list; first beat is at first onset
            beats = [start_time]
            # Index of the tempo we're using
            n = 0
            # Move past all the tempo changes up to the supplied start time
            while beats[-1] > tempo_change_times[n]:
                n += 1
            # Add beats in
            while beats[-1] < note_list[-1].start:
                # Compute expected beat location, one period later
                next_beat = beats[-1] + 60.0/tempii[n]
                # If the beat location passes a tempo change boundary...
                if n < tempo_change_times.shape[0] - 1 and next_beat > tempo_change_times[n + 1]:
                    # Start by setting the beat location to the current beat...
                    next_beat = beats[-1]
                    # with the entire beat remaining
                    beat_remaining = 1.0
                    # While a beat with the current tempo would pass a tempo change boundary...
                    while n < tempo_change_times.shape[0] - 1 and \
                    next_beat + beat_remaining*60.0/tempii[n] >= tempo_change_times[n + 1]:
                        # Compute the extent to which the beat location overshoots
                        overshot_ratio = (tempo_change_times[n + 1] - next_beat)/(60.0/tempii[n])
                        # Add in the amount of the beat during this tempo
                        next_beat += overshot_ratio*60.0/tempii[n]
                        # Less of the beat remains now
                        beat_remaining -= overshot_ratio
                        # Increment the tempo index
                        n = n + 1
                    next_beat += beat_remaining*60./tempii[n]
                beats.append(next_beat)
            return np.array(beats)
        
        # List of possible beat trackings
        beat_candidates = []
        onset_index = 0
        # Try the first 10 (unique) onsets as beat tracking start locations
        while len(beat_candidates) <= 10 and len(beat_candidates) <= len(note_list):
            # Make sure we are using a new start location
            if onset_index == 0 or np.abs(note_list[onset_index - 1].start - note_list[onset_index].start) > .001:
                beat_candidates.append(beat_track_using_tempo(note_list[onset_index].start))
            onset_index += 1
        # Compute onset scores
        onset_scores = np.zeros(len(beat_candidates))
        # Synthesize note onset signal, with velocity-valued spikes at onset times
        fs = 1000
        onset_signal = np.zeros(int(fs*(self.get_end_time() + 1)))
        for note in note_list:
            onset_signal[int(note.start*fs)] += note.velocity
        for n, beats in enumerate(beat_candidates):
            # Create a synthetic beat signal with 25ms windows
            beat_signal = np.zeros(int(fs*(self.get_end_time() + 1)))
            for beat in np.append(0, beats):
                if beat - .025 < 0:
                    beat_signal[:int((beat + .025)*fs)] = np.ones(int(fs*.05 + (beat - 0.025)*fs))
                else:
                    beat_signal[int((beat - .025)*fs):int((beat - .025)*fs) + int(fs*.05)] = np.ones(int(fs*.05))
            # Compute their dot product and normalize to get score
            onset_scores[n] = np.dot(beat_signal, onset_signal)/beats.shape[0]
        # Return the best-scoring beat tracking
        return beat_candidates[np.argmax(onset_scores)]
    
    def get_onsets(self):
        '''
        Return a list of the times of all onsets of all notes from all instruments.
        
        Output:
            onsets - np.ndarray of onset locations, in seconds
        '''
        onsets = np.array([])
        # Just concatenate onsets from all the instruments
        for instrument in self.instruments:
            onsets = np.append( onsets, instrument.get_onsets() )
        # Return them sorted (because why not?)
        return np.sort( onsets )
    
    def get_piano_roll(self, times=None):
        '''
        Get the MIDI data in piano roll notation.
        
        Input:
            times - times of the start of each column in the piano roll.
                    Default None which is np.arange(0, event_times.max(), 1/100.0)
        Output:
            piano_roll - piano roll of MIDI data, flattened across instruments, np.ndarray of size 128 x times.shape[0]
        '''
        # Get piano rolls for each instrument
        piano_rolls = [i.get_piano_roll(times=times) for i in self.instruments]
        # Allocate piano roll, # columns is max of # of columns in all piano rolls
        piano_roll = np.zeros( (128, np.max([p.shape[1] for p in piano_rolls])), dtype=np.int16 )
        # Sum each piano roll into the aggregate piano roll
        for roll in piano_rolls:
            piano_roll[:, :roll.shape[1]] += roll
        return piano_roll

    def get_chroma(self, times=None):
        '''
        Get the MIDI data as a sequence of chroma vectors.
        
        Input:
            times - times of the start of each column in the chroma matrix.
                    Default None which is np.arange(0, event_times.max(), 1/1000.0)
        Output:
            chroma - chroma matrix, flattened across instruments, np.ndarray of size 12 x times.shape[0]
        '''
        # First, get the piano roll
        piano_roll = self.get_piano_roll(times=times)
        # Fold into one octave
        chroma_matrix = np.zeros((12, piano_roll.shape[1]))
        for note in range(12):
            chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)
        return chroma_matrix

    def synthesize(self, fs=44100, method=np.sin):
        '''
        Synthesize the pattern using some waveshape.  Ignores drum track.
        
        Input:
            fs - Sampling rate
            method - If a string, use pyfluidsynth where method is the path to a .sf2 file.
                     If a function, synthesize the notes using this periodic function (e.g. np.sin, scipy.signal.square etc)
                     Defaults to np.sin.  If the .sf2 file is not found or pyfluidsynth is not installed, also uses np.sin.
        Output:
            synthesized - Waveform of the MIDI data, synthesized at fs
        '''
        # Get synthesized waveform for each instrument
        waveforms = [i.synthesize(fs=fs, method=method) for i in self.instruments]
        # Allocate output waveform, with #sample = max length of all waveforms
        synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
        # Sum all waveforms in
        for waveform in waveforms:
            synthesized[:waveform.shape[0]] += waveform
        # Normalize
        synthesized /= np.abs(synthesized).max()
        return synthesized

    def _time_to_tick(self, time):
        '''
        Converts from a time, in seconds, to absolute tick using self.tick_scales
        
        Input:
            time - time, in seconds, of the event
        Output:
            tick - tick - an integer
        '''
        # Ticks will be accumulated over tick scale changes
        tick = 0
        # Iterate through all the tempo changes (tick scale changes!)
        for change_tick, tick_scale in reversed(self.tick_scales):
            change_time = self.tick_to_time[change_tick]
            if time > change_time:
                tick += (time - change_time)/tick_scale
                time = change_time
        return int(tick)
    
    def write(self, filename):
        '''
        Write the PrettyMIDI object out to a .mid file
        
        Input:
            filename - Path to write .mid file to
        '''
        # Initialize list of tracks to output
        tracks = []
        # Create track 0 with timing information
        timing_track = midi.Track()
        # Not sure if time signature is actually necessary
        timing_track += [midi.TimeSignatureEvent(tick=0, data=[4, 2, 24, 8])]
        # Add in each tempo change event
        for (tick, tick_scale) in self.tick_scales:
            tempo_event = midi.SetTempoEvent(tick=tick)
            # Compute the BPM
            tempo_event.set_bpm(60.0/(tick_scale*self.resolution))
            timing_track += [tempo_event]
        # Add in an end of track event
        timing_track += [midi.EndOfTrackEvent(tick=timing_track[-1].tick + 1)]
        tracks += [timing_track]
        # Create a list of possible channels to assign - this seems to matter for some synths.
        channels = range(16)
        # Don't assign the drum channel by mistake!
        channels.remove(9)
        for n, instrument in enumerate(self.instruments):
            # Initialize track for this instrument
            track = midi.Track()
            # If it's a drum event, we need to set channel to 9
            if instrument.is_drum:
                channel = 9
            # Otherwise, choose a channel from the possible channel list
            else:
                channel = channels[n % len(channels)]
            # Set the program number
            program_change = midi.ProgramChangeEvent(tick=0)
            program_change.set_value(instrument.program)
            program_change.channel = channel
            track += [program_change]
            # Add all note events
            for note in instrument.events:
                # Construct the note-on event
                note_on = midi.NoteOnEvent(tick=self._time_to_tick(note.start))
                note_on.set_pitch(note.pitch)
                note_on.set_velocity(note.velocity)
                note_on.channel = channel
                # Also need a note-off event (note on with velocity 0)
                note_off = midi.NoteOnEvent(tick=self._time_to_tick(note.end))
                note_off.set_pitch(note.pitch)
                note_off.set_velocity(0)
                note_off.channel = channel
                # Add notes to track
                track += [note_on, note_off]
            # Add all pitch bend events
            for (time, bend) in instrument.pitch_changes:
                bend_event = midi.PitchWheelEvent(tick=self._time_to_tick(time))
                bend_event.set_pitch(int((bend/2)*8192))
                bend_event.channel = channel
                track += [bend_event]
            # Need to sort all the events by tick time before converting to relative
            tick_sort = np.argsort([event.tick for event in track])
            track = midi.Track([track[n] for n in tick_sort])
            # Finally, add in an end of track event
            track += [midi.EndOfTrackEvent(tick=track[-1].tick + 1)]
            # Add to the list of output tracks
            tracks += [track]
        # Construct an output pattern with the currently stored resolution
        output_pattern = midi.Pattern(resolution=self.resolution, tracks=tracks)
        # Turn ticks to relative, it doesn't work otherwise
        output_pattern.make_ticks_rel()
        # Write it out
        midi.write_midifile(filename, output_pattern)

# <codecell>

class Instrument(object):
    '''
    Object to hold event information for a single instrument
    
    Members:
        program - The program number of this instrument.
        is_drum - Is the instrument a drum instrument (channel 9)?
        events - List of Note objects
        pitch_changes - List of pitch adjustments, in semitones (via the pitch wheel).
                        Tuples of (absolute time, relative pitch adjustment)
    '''
    def __init__(self, program, is_drum=False):
        '''
        Create the Instrument.  events gets initialized to empty list, fill with (Instrument).events.append( event )
        
        Input:
            program - MIDI program number (instrument index)
            is_drum - Is the instrument a drum instrument (channel 9)? Default False
        '''
        self.program = program
        self.is_drum = is_drum
        self.events = []
        self.pitch_changes = []
    
    def get_onsets(self):
        '''
        Get all onsets of all notes played by this instrument.
        
        Output:
            onsets - np.ndarray of all onsets
        '''
        onsets = []
        # Get the note-on time of each note played by this instrument
        for note in self.events:
            onsets.append( note.start )
        # Return them sorted (because why not?)
        return np.sort( onsets )
    
    def get_piano_roll(self, times=None):
        '''
        Get a piano roll notation of the note events of this instrument.
        
        Input:
            times - times of the start of each column in the piano roll, 
                    Default None which is np.arange(0, event_times.max(), 1/100.0)
        Output:
            piano_roll - Piano roll matrix, np.ndarray of size 128 x times.shape[0]
        '''
        # If there are no events, return an empty matrix
        if self.events == []:
            return np.array([[]]*128)
        # Get the end time of the last event
        end_time = np.max([note.end for note in self.events])
        # Sample at 100 Hz
        fs = 100
        # Allocate a matrix of zeros - we will add in as we go
        piano_roll = np.zeros((128, fs*end_time), dtype=np.int16)
        # Drum tracks don't have pitch, so return a matrix of zeros
        if self.is_drum:
            if times is None:
                return piano_roll
            else:
                return np.zeros((128, times.shape[0]), dtype=np.int16)
        # Add up piano roll matrix, note-by-note
        for note in self.events:
            # Should interpolate
            piano_roll[note.pitch, int(note.start*fs):int(note.end*fs)] += note.velocity

        # Process pitch changes
        for ((start, bend), (end, _)) in zip( self.pitch_changes, self.pitch_changes[1:] + [(end_time, 0)] ):
            # Piano roll is already generated with everything bend = 0
            if np.abs( bend ) < 1/8192.0:
                continue
            # Get integer and decimal part of bend amount
            bend_int = int( np.sign( bend )*np.floor( np.abs( bend ) ) )
            bend_decimal = np.abs( bend - bend_int )
            # Construct the bent part of the piano roll
            bent_roll = np.zeros( (128, int(end*fs) - int(start*fs)) )
            # Easiest to process differently depending on bend sign
            if bend >= 0:
                # First, pitch shift by the int amount
                if bend_int is not 0:
                    bent_roll[bend_int:] = piano_roll[:-bend_int, int(start*fs):int(end*fs)]
                else:
                    bent_roll = piano_roll[:, int(start*fs):int(end*fs)]
                # Now, linear interpolate by the decimal place
                bent_roll[1:] = (1 - bend_decimal)*bent_roll[1:] + bend_decimal*bent_roll[:-1]
            else:
                # Same procedure as for positive bends
                if bend_int is not 0:
                    bent_roll[:bend_int] = piano_roll[-bend_int:, int(start*fs):int(end*fs)]
                else:
                    bent_roll = piano_roll[:, int(start*fs):int(end*fs)]
                bent_roll[:-1] = (1 - bend_decimal)*bent_roll[:-1] + bend_decimal*bent_roll[1:]
            # Store bent portion back in piano roll
            piano_roll[:, int(start*fs):int(end*fs)] = bent_roll
        
        if times is None:
            return piano_roll
        piano_roll_integrated = np.zeros((128, times.shape[0]), dtype=np.int16)
        # Convert to column indices
        times = times*fs
        for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
            # Each column is the mean of the columns in piano_roll
            piano_roll_integrated[:, n] = np.mean(piano_roll[:, start:end], axis=1)
        return piano_roll_integrated

    def get_chroma(self, times=None):
        '''
        Get a chroma matrix for the note events in this instrument.
        
        Input:
            times - times of the start of each column in the chroma matrix,
                    Default None which is np.arange(0, event_times.max(), 1/1000.0)
        Output:
            chroma - chroma matrix, np.ndarray of size 12 x times.shape[0]
        '''
        # First, get the piano roll
        piano_roll = self.get_piano_roll(times=times)
        # Fold into one octave
        chroma_matrix = np.zeros((12, piano_roll.shape[1]))
        for note in range(12):
            chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)
        return chroma_matrix

    def synthesize(self, fs=44100, method=np.sin):
        '''
        Synthesize the instrument's notes using some waveshape.  For drum instruments, returns zeros.
        
        Input:
            fs - Sampling rate
            method - If a string, use pyfluidsynth where method is the path to a .sf2 file.
                     If a function, synthesize the notes using this periodic function (e.g. np.sin, scipy.signal.square etc)
                     Defaults to np.sin.  If the .sf2 file is not found or pyfluidsynth is not installed, also uses np.sin.
        Output:
            synthesized - Waveform of the MIDI data, synthesized at fs.  Not normalized!
        '''
        # Pre-allocate output waveform
        synthesized = np.zeros(int(fs*(max([n.end for n in self.events] + [bend[0] for bend in self.pitch_changes]) + 1)))
        # If we're a percussion channel, just return the zeros - can't get FluidSynth to work
        if self.is_drum:
            return synthesized

        # If method is a string and we have fluidsynth, try to use fluidsynth
        if _HAS_FLUIDSYNTH and type(method) == str and os.path.exists(method):
            # Create fluidsynth instance
            fl = fluidsynth.Synth(samplerate=fs)
            # Load in the soundfont
            sfid = fl.sfload(method)
            # Set the channel to 10 if it's a drum channel, 0 otherwise (doesn't actually work)
            channel = 9 if self.is_drum else 0
            # Select the program number
            fl.program_select(channel, sfid, 0, self.program)
            # Collect all notes in one list
            event_list = []
            for note in self.events:
                event_list += [[note.start, 'note on', note.pitch, note.velocity]]
                event_list += [[note.end, 'note off', note.pitch]]
            for bend in self.pitch_changes:
                event_list += [[bend[0], 'pitch bend', bend[1]]]
            # Sort the event list by time
            event_list.sort(key=lambda x: x[0])
            # Add some silence at the beginning according to the time of the first event
            current_sample = int(fs*event_list[0][0])
            # Convert absolute secons to relative samples
            next_event_times = [e[0] for e in event_list[1:]]
            for event, end in zip(event_list[:-1], next_event_times):
                event[0] = int(fs*(end - event[0]))
            event_list[-1][0] = int(fs)
            # Iterate over all events
            for event in event_list:
                # Process events based on type
                if event[1] == 'note on':
                    fl.noteon(channel, event[2], event[3])
                elif event[1] == 'note off':
                    fl.noteoff(channel, event[2])
                elif event[1] == 'pitch bend':
                    fl.pitch_bend(channel, int(8192*(event[2]/2)))
                # Add in these samples
                synthesized[current_sample:current_sample + event[0]] += fl.get_samples(event[0])[::2]
                # Increment the current sample
                current_sample += event[0]
            # Close fluidsynth
            fl.delete()
            
        # If method is a function, use it to synthesize (also a failure mode for the above)
        else:
            # If the above if statement failed, we need to revert back to default
            if not hasattr(method, '__call__'):
                warnings.warn("fluidsynth was requested, but the .sf2 file was not found or pyfluidsynth is not installed.",
                              RuntimeWarning)
                method = np.sin
            # This is a simple way to make the end of the notes fade-out without clicks
            fade_out = np.linspace( 1, 0, .1*fs )
            # Add in waveform for each note
            for note in self.events:
                # Indices in samples of this note
                start = int(fs*note.start)
                end = int(fs*note.end)
                # Get frequency of note from MIDI note number
                frequency = 440*(2.0**((note.pitch - 69)/12.0))
                # Synthesize using wave function at this frequency
                note_waveform = method(2*np.pi*frequency*np.arange(end - start)/fs)
                # Apply an exponential envelope
                envelope = np.exp(-np.arange(end - start)/(1.0*fs))
                # Make the end of the envelope be a fadeout
                if envelope.shape[0] > fade_out.shape[0]:
                    envelope[-fade_out.shape[0]:] *= fade_out
                else:
                    envelope *= np.linspace( 1, 0, envelope.shape[0] )
                # Multiply by velocity (don't think it's linearly scaled but whatever)
                envelope *= note.velocity
                # Add in envelope'd waveform to the synthesized signal
                synthesized[start:end] += envelope*note_waveform

        return synthesized
    
    def __repr__(self):
        return 'Instrument(program={}, is_drum={})'.format(self.program, self.is_drum, len(self.events))
        

# <codecell>

class Note(object):
    '''
    A note event.
    
    Members:
        velocity - Note velocity
        pitch - Note pitch, as a MIDI note number
        start - Note on time, absolute, in seconds
        end - Note off time, absolute, in seconds
    '''
    def __init__(self, velocity, pitch, start, end):
        '''
        Create a note object.  pitch_changes is initialized to [], add pitch changes via (Note).pitch_changes.append
        
        Input:
            velocity - Note velocity
            pitch - Note pitch, as a MIDI note number
            start - Note on time, absolute, in seconds
            end - Note off time, absolute, in seconds
        '''
        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end
    
    def __repr__(self):
        return 'Note(start={:f}, end={:f}, pitch={}, velocity={})'.format(self.start, self.end, self.pitch, self.velocity)

