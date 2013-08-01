# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Utility functions for handling MIDI data in an easy to read/manipulate format
'''

# <codecell>

import midi
import numpy as np

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
        # Initialize empty list of instruments
        instruments = []
        
        # Convert tick values in midi_data to absolute, a useful thing.
        midi_data.make_ticks_abs()
        
        # Store the resolution for later use
        self.resolution = midi_data.resolution
        
        # Populate the list of tempo changes (tick scales
        self._get_tempo_changes(midi_data)
        # Check that there are no tempo change events on any tracks other than track 0
        if sum([[event.name == 'Set Tempo' for event in track] for track in midi_data[1:]]):
            print "Warning - tempo change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file.  Timing may be wrong."
            
        # Populate teh list of instruments
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
    
    def _tick_to_time(self, tick):
        '''
        Converts from a tick to a time, in seconds, using self.tick_scales
        
        Input:
            tick - absolute tick
        Output:
            time - time, in seconds, of the tick
        '''
        # Time will be accumulated over tick scale changes
        time = 0
        # Iterate through all the tempo changes (tick scale changes!)
        for change_tick, tick_scale in reversed(self.tick_scales):
            if tick > change_tick:
                time += tick_scale*(tick - change_tick)
                tick = change_tick
        return time
    
    def _get_instruments(self, midi_data):
        '''
        Populates the list of instruments in midi_data.
        
        Input:
            midi_data - midi.FileReader object
        '''
        

    def get_tempii(self):
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
            tempo_change_times[n] = self._tick_to_time(tick)
            # Convert tick scale to a tempo
            tempii[n] = 60.0/(tick_scale*self.resolution)
        return tempo_change_times, tempii
        
    
    def get_beats(self):
        '''
        Return a list of (probably correct) beat locations in the MIDI file
        
        Output:
            beats - np.ndarray of beat locations, in seconds
        '''
    
    def get_onsets(self):
        '''
        Return a list of the times of all onsets of all notes from all instruments.
        
        Output:
            onsets - np.ndarray of onset locations, in seconds
        '''
    
    def get_piano_roll(self, times=None):
        '''
        Get the MIDI data in piano roll notation.
        
        Input:
            times - times of the start of each column in the piano roll, default None which is np.arange(0, event_times.max(), 1/1000.0)
        Output:
            piano_roll - piano roll of MIDI data, flattened across instruments, np.ndarray of size 127 x times.shape[0]
        '''

    def get_chroma(self, times=None):
        '''
        Get the MIDI data as a sequence of chroma vectors.
        
        Input:
            times - times of the start of each column in the chroma matrix, default None which is np.arange(0, event_times.max(), 1/1000.0)
        Output:
            piano_roll - chroma matrix, flattened across instruments, np.ndarray of size 127 x times.shape[0]
        '''

# <codecell>

class Instrument(object):
    '''
    Object to hold event information for a single instrument
    
    Members:
        program - The program number of this instrument.
        is_drum - Is the instrument a drum instrument (channel 9)?
        events - List of Note objects
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
    
    def get_onsets(self):
        '''
        Get all onsets of all notes played by this instrument.
        
        Output:
            onsets - np.ndarray of all onsets
        '''
    
    def get_piano_roll(self, times=None):
        '''
        Get a piano roll notation of the note events of this instrument.
        
        Input:
            times - times of the start of each column in the piano roll, default None which is np.arange(0, event_times.max(), 1/1000.0)
        Output:
            piano_roll - Piano roll matrix, np.ndarray of size 127 x times.shape[0]
        '''

    def get_chroma(self, times=None):
        '''
        Get a chroma matrix for the note events in this instrument.
        
        Input:
            times - times of the start of each column in the chroma matrix, default None which is np.arange(0, event_times.max(), 1/1000.0)
        Output:
            piano_roll - chroma matrix, np.ndarray of size 127 x times.shape[0]
        '''

# <codecell>

class Note(object):
    '''
    A note event.
    
    Members:
        velocity - Note velocity
        pitch - Note pitch, as a MIDI note number
        start - Note on time, absolute, in seconds
        end - Note off time, absolute, in seconds
        pitch_changes - List of pitch adjustments, in semitones (via the pitch wheel).  Tuples of (absolute time, relative pitch adjustment)
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
        self.pitch_changes = []

