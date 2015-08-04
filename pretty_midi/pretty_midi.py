"""Utility functions for handling MIDI data in an easy to read/manipulate
format

"""

import midi
import numpy as np

try:
    import fluidsynth
    _HAS_FLUIDSYNTH = True
except ImportError:
    _HAS_FLUIDSYNTH = False

import os
import warnings
import pkg_resources
import re
import collections
import copy

DEFAULT_SF2 = 'TimGM6mb.sf2'

# The largest we'd ever expect a tick to be
MAX_TICK = 1e7


class PrettyMIDI(object):
    """A container for MIDI data in an easily-manipulable format.

    Parameters
    ----------
    midi_file : str or file
        Path or file pointer to a MIDI file.
        Default None which means create an empty class with the supplied values
        for resolutiona and initial tempo.
    resolution : int
        Resolution of the MIDI data, when no file is provided.
    intitial_tempo : float
        Initial tempo for the MIDI data, when no file is provided.

    Attributes
    ----------
    instruments : list
        List of pretty_midi.Instrument objects.

    """

    def __init__(self, midi_file=None, resolution=220, initial_tempo=120.):
        """Initialize the PrettyMIDI container, either by populating it with
        MIDI data from a file or from scratch with no data.

        """
        if midi_file is not None:
            # Load in the MIDI data using the midi module
            midi_data = midi.read_midifile(midi_file)

            # Convert tick values in midi_data to absolute, a useful thing.
            midi_data.make_ticks_abs()

            # Store the resolution for later use
            self.resolution = midi_data.resolution

            # populate the list of tempo changes (tick scales)
            self._load_tempo_changes(midi_data)

            # Update the array which maps ticks to time
            max_tick = max([max([e.tick for e in t]) for t in midi_data]) + 1
            # If max_tick is huge, the MIDI file is probably corrupt
            # and creating the __tick_to_time array will thrash memory
            if max_tick > MAX_TICK:
                raise ValueError(('MIDI file has a largest tick of {},'
                                  ' it is likely corrupt'.format(max_tick)))

            # Create list that maps ticks to time in seconds
            self._update_tick_to_time(max_tick)

            # Populate the list of key and time signature changes
            self._load_metadata(midi_data)

            # Check that there are only tempo change events on track 0
            if sum([sum([event.name == 'Set Tempo' for event in track])
                    for track in midi_data[1:]]):
                warnings.warn(("Tempo change events found on non-zero tracks."
                               "  This is not a valid type 0 or type 1 MIDI "
                               "file.  Timing may be wrong."), RuntimeWarning)

            # Check that there are only key signature change events on track 0
            if sum([sum([isinstance(event, midi.events.KeySignatureEvent)
                        for event in track]) for track in midi_data[1:]]):
                warnings.warn(("Key signature change events"
                               " found on non-zero tracks."
                               "  This is not a valid type 0 or type 1 MIDI "
                               "file.  Key Signature may be wrong."),
                              RuntimeWarning)

            # Check that there are only time signature change events on track 0
            if sum([sum([isinstance(event, midi.events.TimeSignatureEvent)
                        for event in track]) for track in midi_data[1:]]):
                warnings.warn(("Time signature change events"
                               " found on non-zero tracks."
                               "  This is not a valid type 0 or type 1 MIDI "
                               "file.  Time Signature may be wrong."),
                              RuntimeWarning)

            # Populate the list of instruments
            self._load_instruments(midi_data)

        else:
            self.resolution = resolution
            # Compute the tick scale for the provided initial tempo
            # and let the tick scale start from 0
            self.__tick_scales = [(0, 60.0/(initial_tempo*self.resolution))]
            # Only need to convert one tick to time
            self.__tick_to_time = [0]
            # Empty instruments list
            self.instruments = []

    def _load_tempo_changes(self, midi_data):
        """Populates self.__tick_scales with tuples of (tick, tick_scale)

        Parameters
        ----------
            - midi_data : midi.FileReader
                MIDI object from which data will be read
        """

        # MIDI data is given in "ticks".
        # We need to convert this to clock seconds.
        # The conversion factor involves the BPM, which may change over time.
        # So, create a list of tuples, (time, tempo)
        # denoting a tempo change at a certain time.
        # By default, set the tempo to 120 bpm, starting at time 0
        self.__tick_scales = [(0, 60.0/(120.0*midi_data.resolution))]
        # For SMF file type 0, all events are on track 0.
        # For type 1, all tempo events should be on track 1.
        # Everyone ignores type 2.
        # So, just look at events on track 0
        for event in midi_data[0]:
            if isinstance(event, midi.events.SetTempoEvent):
                # Only allow one tempo change event at the beginning
                if event.tick == 0:
                    bpm = event.get_bpm()
                    self.__tick_scales = [(0, 60.0/(bpm*midi_data.resolution))]
                else:
                    # Get time and BPM up to this point
                    _, last_tick_scale = self.__tick_scales[-1]
                    tick_scale = 60.0/(event.get_bpm()*midi_data.resolution)
                    # Ignore repetition of BPM, which happens often
                    if tick_scale != last_tick_scale:
                        self.__tick_scales.append((event.tick, tick_scale))

    def _load_metadata(self, midi_data):
        """Populates the lists of tempo key and time signature changes

        Populates self.__time_signatures with TimeSignature objects.
            It can be accessed by the get_time_signatures method.
        Populates self.__key_changes with KeySignature objects.
            It can be accessed by the get_key_signatures method.

        Parameters
        ----------
            - midi_data : midi.FileReader
                MIDI object from which data will be read
        """

        # list to store key signature changes
        self.__key_changes = []

        # list to store time signatures changes
        self.__time_signatures = []

        for event in midi_data[0]:
            if isinstance(event, midi.events.KeySignatureEvent):
                key_obj = KeySignature(midi_key_to_key_name(event),
                              self.__tick_to_time[event.tick])
                self.__key_changes.append(key_obj)

            elif isinstance(event, midi.events.TimeSignatureEvent):
                ts_obj = TimeSignature(event.get_numerator(),
                                       event.get_denominator(),
                                       self.__tick_to_time[event.tick])
                self.__time_signatures.append(ts_obj)

    def _update_tick_to_time(self, max_tick):
        """Creates __tick_to_time, a class member array which maps ticks to time
        starting from tick 0 and ending at max_tick

        Parameters
        ----------
        max_tick : int
            last tick to compute time for

        """
        # Allocate tick to time array - indexed by tick from 0 to max_tick
        self.__tick_to_time = np.zeros(max_tick + 1)
        # Keep track of the end time of the last tick in the previous interval
        last_end_time = 0
        # Cycle through intervals of different tempii
        for (start_tick, tick_scale), (end_tick, _) in \
                zip(self.__tick_scales[:-1], self.__tick_scales[1:]):
            # Convert ticks in this interval to times
            ticks = np.arange(end_tick - start_tick + 1)
            self.__tick_to_time[start_tick:end_tick + 1] = (last_end_time +
                                                            tick_scale*ticks)
            # Update the time of the last tick in this interval
            last_end_time = self.__tick_to_time[end_tick]
        # For the final interval, use the final tempo setting
        # and ticks from the final tempo setting until max_tick
        start_tick, tick_scale = self.__tick_scales[-1]
        ticks = np.arange(max_tick + 1 - start_tick)
        self.__tick_to_time[start_tick:] = (last_end_time +
                                            tick_scale*ticks)

    def _load_instruments(self, midi_data):
        """Populates the list of instruments in midi_data.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read

        """

        # Initialize empty list of instruments
        self.instruments = []
        for track in midi_data:
            # Keep track of last note on location:
            # key = (instrument, is_drum, note),
            # value = (note on time, velocity)
            last_note_on = collections.defaultdict(list)
            # Keep track of which instrument is playing in each channel
            # initialize to program 0 for all channels
            current_instrument = np.zeros(16, dtype=np.int)
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
                    note_on_index = (current_instrument[event.channel],
                                     is_drum, event.pitch)
                    last_note_on[note_on_index].append((
                        self.__tick_to_time[event.tick],
                        event.velocity))
                # Note offs can also be note on events with 0 velocity
                elif event.name == 'Note Off' or (event.name == 'Note On'
                                                  and event.velocity == 0):
                    # Get the instrument's drum type
                    is_drum = (event.channel == 9)
                    # Check that a note-on exists (ignore spurious note-offs)
                    if (current_instrument[event.channel],
                            is_drum, event.pitch) in last_note_on:
                        # Get the start/stop times and velocity of every note
                        # which was turned on with this instrument/drum/pitch
                        for start, velocity in last_note_on[
                            (current_instrument[event.channel],
                             is_drum, event.pitch)]:
                            end = self.__tick_to_time[event.tick]
                            # Create the note event
                            note = Note(velocity, event.pitch, start, end)
                            # Get the program and drum type for the current
                            # instrument
                            program = current_instrument[event.channel]
                            # Retrieve the Instrument instance for the current
                            # instrument
                            instrument = self.__get_instrument(program,
                                                               is_drum)
                            # Add the note event
                            instrument.notes.append(note)
                        # Remove the last note on for this instrument
                        del last_note_on[(current_instrument[event.channel],
                                          is_drum, event.pitch)]
                # Store pitch bends
                elif event.name == 'Pitch Wheel':
                    # Create pitch bend class instance
                    bend = PitchBend(event.pitch,
                                     self.__tick_to_time[event.tick])
                    # Get the program and drum type for the current inst
                    program = current_instrument[event.channel]
                    is_drum = (event.channel == 9)
                    # Retrieve the Instrument instance for the current inst
                    instrument = self.__get_instrument(program, is_drum)
                    # Add the pitch bend event
                    instrument.pitch_bends.append(bend)
                # Store control changes
                elif event.name == 'Control Change':
                    control_change = ControlChange(
                        event.data[0], event.data[1],
                        self.__tick_to_time[event.tick])
                    # Get the program and drum type for the current inst
                    program = current_instrument[event.channel]
                    is_drum = (event.channel == 9)
                    # Retrieve the Instrument instance for the current inst
                    instrument = self.__get_instrument(program, is_drum)
                    # Add the control change event
                    instrument.control_changes.append(control_change)

    def __get_instrument(self, program, is_drum):
        """Gets the Instrument corresponding to the given program number and
        drum/non-drum type.  If no such instrument exists, one is created.

        """
        for instrument in self.instruments:
            if (instrument.program == program
                    and instrument.is_drum == is_drum):
                # Add this note event
                return instrument
        # Create the instrument if none was found
        self.instruments.append(Instrument(program, is_drum))
        instrument = self.instruments[-1]
        return instrument

    def get_tempo_changes(self):
        """Return arrays of tempo changes and their times.

        This is direct from the MIDI file.

        Returns
        -------
        tempo_change_times : np.ndarray
            Times, in seconds, where the tempo changes.
        tempii : np.ndarray
            What the tempo is at each point in time in tempo_change_times

        """

        # Pre-allocate return arrays
        tempo_change_times = np.zeros(len(self.__tick_scales))
        tempii = np.zeros(len(self.__tick_scales))
        for n, (tick, tick_scale) in enumerate(self.__tick_scales):
            # Convert tick of this tempo change to time in seconds
            tempo_change_times[n] = self.__tick_to_time[tick]
            # Convert tick scale to a tempo
            tempii[n] = 60.0/(tick_scale*self.resolution)
        return tempo_change_times, tempii

    def get_end_time(self):
        """Returns the time of the end of this MIDI file (latest note-off event).

        Returns
        -------
        end_time : float
            Time, in seconds, where this MIDI file ends

        """
        # Cycle through all notes from all instruments and find the largest
        events = ([n.end for i in self.instruments for n in i.notes] +
                  [b.time for i in self.instruments for b in i.pitch_bends])
        # If there are no events, return 0
        if len(events) == 0:
            return 0.
        else:
            return max(events)

    def estimate_tempii(self):
        """Return an empirical estimate of tempos in the piece and each tempo's
        probability.
        Based on "Automatic Extraction of Tempo and Beat from Expressive
        Performance", Dixon 2001

        Returns
        -------
        tempos : np.ndarray
            Array of estimated tempos, in bpm
        probabilities : np.ndarray
            Array of the probability of each tempo estimate

        """
        # Grab the list of onsets
        onsets = self.get_onsets()
        # Compute inner-onset intervals
        ioi = np.diff(onsets)
        # "Rhythmic information is provided by IOIs in the range of
        # approximately 50ms to 2s (Handel, 1989)"
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
                clusters[k] = (cluster_counts[k]*clusters[k]
                               + interval)/(cluster_counts[k] + 1)
                # Update number of elements in cluster
                cluster_counts[k] += 1
            # No cluster is close, make a new one
            else:
                clusters = np.append(clusters, interval)
                cluster_counts = np.append(cluster_counts, 1.)
        # Sort the cluster list by count
        cluster_sort = np.argsort(cluster_counts)[::-1]
        clusters = clusters[cluster_sort]
        cluster_counts = cluster_counts[cluster_sort]
        # Normalize the cluster scores
        cluster_counts /= cluster_counts.sum()
        return 60./clusters, cluster_counts

    def estimate_tempo(self):
        """Returns the best tempo estimate from estimate_tempii(), for
        convenience

        Returns
        -------
        tempo : float
            Estimated tempo, in bpm

        """
        return self.estimate_tempii()[0][0]

    def get_beats(self):
        """Return a list of beat locations, estimated according to the MIDI
        file tempo changes.

        Will not be correct if the MIDI data has been modified without changing
        tempo information.

        Returns
        -------
        beats : np.ndarray
            Beat locations, in seconds

        """
        # Get a sorted list of all notes from all instruments
        note_list = [n for i in self.instruments for n in i.notes]
        note_list.sort(key=lambda note: note.start)
        # Get tempo changes and tempos
        tempo_change_times, tempii = self.get_tempo_changes()

        def beat_track_using_tempo(start_time):
            """Starting from start_time, place beats according to the MIDI
            file's designated tempo changes.

            """
            # Create beat list; first beat is at first onset
            beats = [start_time]
            # Index of the tempo we're using
            n = 0
            # Move past all the tempo changes up to the supplied start time
            while (n < tempo_change_times.shape[0] - 1 and
                   beats[-1] > tempo_change_times[n]):
                n += 1
            # Get track end time
            end_time = self.get_end_time()
            # Add beats in
            while beats[-1] < end_time:
                # Compute expected beat location, one period later
                next_beat = beats[-1] + 60.0/tempii[n]
                # If the beat location passes a tempo change boundary...
                if (n < tempo_change_times.shape[0] - 1 and
                        next_beat > tempo_change_times[n + 1]):
                    # Start by setting the beat location to the current beat...
                    next_beat = beats[-1]
                    # with the entire beat remaining
                    beat_remaining = 1.0
                    # While a beat with the current tempo would pass a tempo
                    # change boundary...
                    while (n < tempo_change_times.shape[0] - 1 and
                           next_beat + beat_remaining*60.0/tempii[n]
                           >= tempo_change_times[n + 1]):
                        # Compute the amount the beat location overshoots
                        overshot_ratio = (tempo_change_times[n + 1]
                                          - next_beat)/(60.0/tempii[n])
                        # Add in the amount of the beat during this tempo
                        next_beat += overshot_ratio*60.0/tempii[n]
                        # Less of the beat remains now
                        beat_remaining -= overshot_ratio
                        # Increment the tempo index
                        n = n + 1
                    next_beat += beat_remaining*60./tempii[n]
                beats.append(next_beat)
            # The last beat will pass the end_time barrier, so don't return it
            return np.array(beats[:-1])

        # List of possible beat trackings
        beat_candidates = []
        onset_index = 0
        # Try the first 10 (unique) onsets as beat tracking start locations
        while (len(beat_candidates) <= 10 and
               len(beat_candidates) <= len(note_list) and
               onset_index < len(note_list)):
            # Make sure we are using a new start location
            if onset_index == 0 or np.abs(note_list[onset_index - 1].start -
                                          note_list[onset_index].start) > .001:
                beat_candidates.append(
                    beat_track_using_tempo(note_list[onset_index].start))
            onset_index += 1
        # Compute onset scores
        onset_scores = np.zeros(len(beat_candidates))
        # Synthesize note onset signal, with velocity-valued spikes at onsets
        fs = 1000
        onset_signal = np.zeros(int(fs*(self.get_end_time() + 1)))
        for note in note_list:
            onset_signal[int(note.start*fs)] += note.velocity
        for n, beats in enumerate(beat_candidates):
            # Create a synthetic beat signal with 25ms windows
            beat_signal = np.zeros(int(fs*(self.get_end_time() + 1)))
            for beat in np.append(0, beats):
                if beat - .025 < 0:
                    beat_window = np.ones(int(fs*.05 + (beat - 0.025)*fs))
                    beat_signal[:int((beat + .025)*fs)] = beat_window
                else:
                    beat_start = int((beat - .025)*fs)
                    beat_end = beat_start + int(fs*.05)
                    beat_window = np.ones(int(fs*.05))
                    beat_signal[beat_start:beat_end] = beat_window
            # Compute their dot product and normalize to get score
            onset_scores[n] = np.dot(beat_signal, onset_signal)/beats.shape[0]
        # Return the best-scoring beat tracking
        return beat_candidates[np.argmax(onset_scores)]

    def get_onsets(self):
        """Return a sorted list of the times of all onsets of all notes from
        all instruments.  May have duplicate entries.

        Returns
        -------
        onsets : np.ndarray
            Onset locations, in seconds

        """
        onsets = np.array([])
        # Just concatenate onsets from all the instruments
        for instrument in self.instruments:
            onsets = np.append(onsets, instrument.get_onsets())
        # Return them sorted (because why not?)
        return np.sort(onsets)

    def get_piano_roll(self, fs=100, times=None, include_pitch_bends=True):
        """Get the MIDI data in piano roll notation.

        Parameters
        ----------
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart
            by 1./fs seconds
        times : np.ndarray
            Times of the start of each column in the piano roll.
            Default None which is np.arange(0, get_end_time(), 1./fs)

        Returns
        -------
        piano_roll : np.ndarray, shape=(128,times.shape[0])
            Piano roll of MIDI data, flattened across instruments

        """

        # If there are no instruments, return an empty array
        if len(self.instruments) == 0:
            return np.zeros((128, 0))

        # Get piano rolls for each instrument
        piano_rolls = [i.get_piano_roll(
                                    fs=fs,
                                    times=times,
                                    include_pitch_bends=include_pitch_bends)
                       for i in self.instruments]
        # Allocate piano roll,
        # number of columns is max of # of columns in all piano rolls
        piano_roll = np.zeros((128, np.max([p.shape[1] for p in piano_rolls])),
                              dtype=np.int16)
        # Sum each piano roll into the aggregate piano roll
        for roll in piano_rolls:
            piano_roll[:, :roll.shape[1]] += roll
        return piano_roll

    def get_chroma(self, fs=100, times=None, include_pitch_bends=True):
        """Get the MIDI data as a sequence of chroma vectors.

        Parameters
        ----------
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart
            by 1./fs seconds
        times : np.ndarray
            Times of the start of each column in the piano roll.
            Default None which is np.arange(0, get_end_time(), 1./fs)

        Returns
        -------
        piano_roll : np.ndarray, shape=(12,times.shape[0])
            Chromagram of MIDI data, flattened across instruments

        """

        # First, get the piano roll
        piano_roll = self.get_piano_roll(
            fs=fs,
            times=times,
            include_pitch_bends=include_pitch_bends)
        # Fold into one octave
        chroma_matrix = np.zeros((12, piano_roll.shape[1]))
        for note in range(12):
            chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)
        return chroma_matrix

    def get_key_changes(self):
        """Returns a list with KeySignature objects acquired directly
        from the midi file.

        Returns
        -------
            self.__key_changes : list
                List with KeySignature objects
        """
        return self.__key_changes

    def get_time_signatures(self):
        """Returns a list with TimeSignature objects acquired directly
        from the midi file.

        Returns
        -------
            self.__time_signatures : list,
                List with TimeSignatures objects
        """
        return self.__time_signatures

    def synthesize(self, fs=44100, wave=np.sin):
        """Synthesize the pattern using some waveshape.  Ignores drum track.

        Parameters
        ----------
        fs : int
            Sampling rate of the synthesized audio signal, default 44100
        wave : function
            Function which returns a periodic waveform,
            e.g. np.sin, scipy.signal.square, etc.  Default np.sin

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at fs

        """
        # If there are no instruments, return an empty array
        if len(self.instruments) == 0:
            return np.array([])
        # Get synthesized waveform for each instrument
        waveforms = [i.synthesize(fs=fs, wave=wave) for i in self.instruments]
        # Allocate output waveform, with #sample = max length of all waveforms
        synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
        # Sum all waveforms in
        for waveform in waveforms:
            synthesized[:waveform.shape[0]] += waveform
        # Normalize
        synthesized /= np.abs(synthesized).max()
        return synthesized

    def fluidsynth(self, fs=44100, sf2_path=None):
        """Synthesize using fluidsynth.

        Parameters
        ----------
        fs : int
            Sampling rate to synthesize
        sf2_path : str
            Path to a .sf2 file.
            Default None, which uses the TimGM6mb.sf2 file included with
            pretty_midi.

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at fs

        """
        # If there are no instruments, return an empty array
        if len(self.instruments) == 0:
            return np.zeros((128, 0))
        # Get synthesized waveform for each instrument
        waveforms = [i.fluidsynth(fs=fs,
                                  sf2_path=sf2_path) for i in self.instruments]
        # Allocate output waveform, with #sample = max length of all waveforms
        synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
        # Sum all waveforms in
        for waveform in waveforms:
            synthesized[:waveform.shape[0]] += waveform
        # Normalize
        synthesized /= np.abs(synthesized).max()
        return synthesized

    def tick_to_time(self, tick):
        """Converts from an absolute tick to time in seconds using
        self.__tick_to_time

        Parameters
        ----------
        tick : int
            absolute tick to convert

        Returns
        -------
        time : float
            time in seconds of tick

        """
        # Check that the tick isn't too big
        if tick >= MAX_TICK:
            raise IndexError('Supplied tick is too large.')
        # If we haven't compute the mapping for a tick this large, compute it
        if tick >= len(self.__tick_to_time):
            self._update_tick_to_time(tick)
        # Ticks should be integers
        if type(tick) != int:
            warnings.warn('tick should be an int.')
        # Otherwise just return the time
        return self.__tick_to_time[int(tick)]

    def time_to_tick(self, time):
        """Converts from a time in seconds to absolute tick using
        self.__tick_scales

        Parameters
        ----------
        time : float
            Time, in seconds

        Returns
        -------
        tick : int
            Absolute tick corresponding to the supplied time

        """
        # Ticks will be accumulated over tick scale changes
        tick = 0
        # Iterate through all the tempo changes (tick scale changes!)
        for change_tick, tick_scale in reversed(self.__tick_scales):
            change_time = self.tick_to_time(change_tick)
            if time > change_time:
                tick += (time - change_time)/tick_scale
                time = change_time
        return int(tick)

    def adjust_times(self, original_times, new_times):
        """Adjusts the timing of the events in the MIDI object.
        The parameters `original_times` and `new_times` define a mapping, so
        that if an event originally occurs at time `original_times[n]`, it
        will be moved so that it occurs at `new_times[n]`.  If events don't
        occur exactly on a time in `original_times`, their timing will be
        linearly interpolated.

        Parameters
        ----------
        original_times : np.ndarray
            Times to map from
        new_times : np.ndarray
            New times to map to

        """
        # Only include notes within start/end time of the provided times
        for instrument in self.instruments:
            valid_notes = []
            for note in instrument.notes:
                if note.start >= original_times[0] and \
                        note.end <= original_times[-1]:
                    valid_notes.append(copy.deepcopy(note))
            instrument.notes = valid_notes
        # Get array of note-on locations and correct them
        note_ons = np.array([note.start for instrument in self.instruments
                             for note in instrument.notes])
        aligned_note_ons = np.interp(note_ons, original_times, new_times)
        # Same for note-offs
        note_offs = np.array([note.end for instrument in self.instruments
                              for note in instrument.notes])
        aligned_note_offs = np.interp(note_offs, original_times, new_times)
        # Same for pitch bends
        pitch_bends = np.array([bend.time for instrument in self.instruments
                                for bend in instrument.pitch_bends])
        aligned_pitch_bends = np.interp(pitch_bends, original_times, new_times)
        ccs = np.array([cc.time for instrument in self.instruments
                        for cc in instrument.control_changes])
        aligned_ccs = np.interp(ccs, original_times, new_times)
        # Correct notes
        for n, note in enumerate([note for instrument in self.instruments
                                  for note in instrument.notes]):
            note.start = (aligned_note_ons[n] > 0)*aligned_note_ons[n]
            note.end = (aligned_note_offs[n] > 0)*aligned_note_offs[n]
        # After performing alignment, some notes may have an end time which is
        # on or before the start time.  Remove these!
        self.remove_invalid_notes()
        # Correct pitch changes
        for n, bend in enumerate([bend for instrument in self.instruments
                                  for bend in instrument.pitch_bends]):
            bend.time = (aligned_pitch_bends[n] > 0)*aligned_pitch_bends[n]
        for n, cc in enumerate([cc for instrument in self.instruments
                                for cc in instrument.control_changes]):
            cc.time = (aligned_ccs[n] > 0)*aligned_ccs[n]

    def remove_invalid_notes(self):
        """Removes any notes which have an end time <= start time.

        """
        # Simply call the child method on all instruments
        for instrument in self.instruments:
            instrument.remove_invalid_notes()

    def write(self, filename):
        """Write the PrettyMIDI object out to a .mid file

        Parameters
        ----------
        filename : str
            Path to write .mid file to

        """
        # Initialize list of tracks to output
        tracks = []
        # Create track 0 with timing information
        timing_track = midi.Track(tick_relative=False)
        # Not sure if time signature is actually necessary
        timing_track += [midi.TimeSignatureEvent(tick=0, data=[4, 2, 24, 8])]
        # Add in each tempo change event
        for (tick, tick_scale) in self.__tick_scales:
            tempo_event = midi.SetTempoEvent(tick=tick)
            # Compute the BPM
            tempo_event.set_bpm(60.0/(tick_scale*self.resolution))
            timing_track += [tempo_event]
        # Add in an end of track event
        timing_track += [midi.EndOfTrackEvent(tick=timing_track[-1].tick + 1)]
        tracks += [timing_track]
        # Create a list of possible channels to assign - this seems to matter
        # for some synths.
        channels = range(16)
        # Don't assign the drum channel by mistake!
        channels.remove(9)
        for n, instrument in enumerate(self.instruments):
            # Initialize track for this instrument
            track = midi.Track(tick_relative=False)
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
            for note in instrument.notes:
                # Construct the note-on event
                note_on = midi.NoteOnEvent(tick=self.time_to_tick(note.start))
                note_on.set_pitch(note.pitch)
                note_on.set_velocity(note.velocity)
                note_on.channel = channel
                # Also need a note-off event (note on with velocity 0)
                note_off = midi.NoteOnEvent(tick=self.time_to_tick(note.end))
                note_off.set_pitch(note.pitch)
                note_off.set_velocity(0)
                note_off.channel = channel
                # Add notes to track
                track += [note_on, note_off]
            # Add all pitch bend events
            for bend in instrument.pitch_bends:
                tick = self.time_to_tick(bend.time)
                bend_event = midi.PitchWheelEvent(tick=tick)
                bend_event.set_pitch(bend.pitch)
                bend_event.channel = channel
                track += [bend_event]
            # Add all control change events
            for control_change in instrument.control_changes:
                tick = self.time_to_tick(control_change.time)
                control_event = midi.ControlChangeEvent(tick=tick)
                control_event.set_control(control_change.number)
                control_event.set_value(control_change.value)
                control_event.channel = channel
                track += [control_event]
            # Sort all the events by tick time before converting to relative
            tick_sort = np.argsort([event.tick for event in track])
            track = midi.Track([track[n] for n in tick_sort],
                               tick_relative=False)
            # If there's a note off event and a note on event with the same
            # tick and pitch, put the note off event first
            for n, (event1, event2) in enumerate(zip(track[:-1], track[1:])):
                if (event1.tick == event2.tick and
                        event1.name == 'Note On' and
                        event2.name == 'Note On' and
                        event1.pitch == event2.pitch and
                        event1.velocity != 0 and
                        event2.velocity == 0):
                    track[n] = event2
                    track[n + 1] = event1
            # Finally, add in an end of track event
            track += [midi.EndOfTrackEvent(tick=track[-1].tick + 1)]
            # Add to the list of output tracks
            tracks += [track]
        # Construct an output pattern with the currently stored resolution
        output_pattern = midi.Pattern(resolution=self.resolution,
                                      tracks=tracks,
                                      tick_relative=False)
        # Turn ticks to relative, it doesn't work otherwise
        output_pattern.make_ticks_rel()
        # Write it out
        midi.write_midifile(filename, output_pattern)


class Instrument(object):
    """Object to hold event information for a single instrument

    Parameters
    ----------
    program : int
        MIDI program number (instrument index), in [0, 127]
    is_drum : bool
        Is the instrument a drum instrument (channel 9)?
        Default False

    Attributes
    ----------
    program : int
        The program number of this instrument.
    is_drum : bool
        Is the instrument a drum instrument (channel 9)?
    notes : list
        List of Note objects
    pitch_bends : list
        List of of PitchBend objects
    control_changes : list
        List of ControlChange objects

    """

    def __init__(self, program, is_drum=False):
        """Create the Instrument.
        notes gets initialized to empty list.
        Fill with `(Instrument).notes.append(event)`

        """
        self.program = program
        self.is_drum = is_drum
        self.notes = []
        self.pitch_bends = []
        self.control_changes = []

    def get_onsets(self):
        """Get all onsets of all notes played by this instrument.
        May contain duplicates

        Returns
        -------
        onsets : np.ndarray
                List of all note onsets

        """
        onsets = []
        # Get the note-on time of each note played by this instrument
        for note in self.notes:
            onsets.append(note.start)
        # Return them sorted (because why not?)
        return np.sort(onsets)

    def get_piano_roll(self, fs=100, times=None, include_pitch_bends=True):
        """Get a piano roll notation of the note events of this instrument.

        Parameters
        ----------
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart
            by 1./fs seconds
        times : np.ndarray
            times of the start of each column in the piano roll,
            Default None which is np.arange(0, get_end_time(), 1./fs)

        Returns
        -------
        piano_roll : np.ndarray, shape=(128, times.shape[0])
            Piano roll matrix of this instrument

        """

        # If there are no notes, return an empty matrix
        if self.notes == []:
            return np.array([[]]*128)
        # Get the end time of the last event
        end_time = self.get_end_time()
        # Extend end time if one was provided
        if times is not None and times[-1] > end_time:
            end_time = times[-1]
        # Allocate a matrix of zeros - we will add in as we go
        piano_roll = np.zeros((128, int(fs*end_time)), dtype=np.int16)
        # Drum tracks don't have pitch, so return a matrix of zeros
        if self.is_drum:
            if times is None:
                return piano_roll
            else:
                return np.zeros((128, times.shape[0]), dtype=np.int16)
        # Add up piano roll matrix, note-by-note
        for note in self.notes:
            # Should interpolate
            piano_roll[note.pitch,
                       int(note.start*fs):int(note.end*fs)] += note.velocity
        if include_pitch_bends:
            # Process pitch changes
            # Need to sort the pitch bend list for the following to work
            ordered_bends = sorted(self.pitch_bends, key=lambda bend: bend.time)
            # Add in a bend of 0 at the end of time
            end_bend = PitchBend(0, end_time)
            for start_bend, end_bend in zip(ordered_bends,
                                            ordered_bends[1:] + [end_bend]):
                # Piano roll is already generated with everything bend = 0
                if np.abs(start_bend.pitch) < 1:
                    continue
                # Get integer and decimal part of bend amount
                start_pitch = pitch_bend_to_semitones(start_bend.pitch)
                bend_int = int(np.sign(start_pitch)*np.floor(np.abs(start_pitch)))
                bend_decimal = np.abs(start_pitch - bend_int)
                # Column indices effected by the bend
                bend_range = np.r_[int(start_bend.time*fs):int(end_bend.time*fs)]
                # Construct the bent part of the piano roll
                bent_roll = np.zeros(piano_roll[:, bend_range].shape)
                # Easiest to process differently depending on bend sign
                if start_bend.pitch >= 0:
                    # First, pitch shift by the int amount
                    if bend_int is not 0:
                        bent_roll[bend_int:] = piano_roll[:-bend_int, bend_range]
                    else:
                        bent_roll = piano_roll[:, bend_range]
                    # Now, linear interpolate by the decimal place
                    bent_roll[1:] = ((1 - bend_decimal)*bent_roll[1:]
                                     + bend_decimal*bent_roll[:-1])
                else:
                    # Same procedure as for positive bends
                    if bend_int is not 0:
                        bent_roll[:bend_int] = piano_roll[-bend_int:, bend_range]
                    else:
                        bent_roll = piano_roll[:, bend_range]
                    bent_roll[:-1] = ((1 - bend_decimal)*bent_roll[:-1]
                                      + bend_decimal*bent_roll[1:])
                # Store bent portion back in piano roll
                piano_roll[:, bend_range] = bent_roll

        if times is None:
            return piano_roll
        piano_roll_integrated = np.zeros((128, times.shape[0]), dtype=np.int16)
        # Convert to column indices
        times = np.array(times*fs, dtype=np.int)
        for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
            # Each column is the mean of the columns in piano_roll
            piano_roll_integrated[:, n] = np.mean(piano_roll[:, start:end],
                                                  axis=1)
        return piano_roll_integrated

    def get_chroma(self, fs=100, times=None, include_pitch_bends=True):
        """Get a chroma matrix for the note events in this instrument.

        Parameters
        ----------
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced
            apart by 1./fs seconds
        times : np.ndarray
            times of the start of each column in the chroma matrix,
            Default None which is np.arange(0, get_end_time(), 1./fs)

        Returns
        -------
        chroma : np.ndarray, shape=(12,times.shape[0])
            Chromagram matrix of this instrument

        """

        # First, get the piano roll
        piano_roll = self.get_piano_roll(
            fs=fs,
            times=times,
            include_pitch_bends=include_pitch_bends)
        # Fold into one octave
        chroma_matrix = np.zeros((12, piano_roll.shape[1]))
        for note in range(12):
            chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)
        return chroma_matrix

    def get_end_time(self):
        """Returns the time of the end of the events in this instrument

        Returns
        -------
        end_time : float
            Time, in seconds, of the end of the last event

        """
        # Cycle through all note ends and all pitch bends and find the largest
        events = ([n.end for n in self.notes] +
                  [b.time for b in self.pitch_bends])
        # If there are no events, just return 0
        if len(events) == 0:
            return 0.
        else:
            return max(events)

    def remove_invalid_notes(self):
        """Removes any notes which have an end time <= start time.

        """
        # Crete a list of all invalid notes
        notes_to_delete = []
        for note in self.notes:
            if note.end <= note.start:
                notes_to_delete.append(note)
        # Remove the notes found
        for note in notes_to_delete:
            self.notes.remove(note)

    def synthesize(self, fs=44100, wave=np.sin):
        """Synthesize the instrument's notes using some waveshape.
        For drum instruments, returns zeros.

        Parameters
        ----------
        fs : int
            Sampling rate of the synthesized audio signal, default 44100
        wave : function
            Function which returns a periodic waveform,
            e.g. np.sin, scipy.signal.square, etc.  Default np.sin

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the instrument's notes, synthesized at fs

        """
        # Pre-allocate output waveform
        synthesized = np.zeros(int(fs*(self.get_end_time() + 1)))

        # If we're a percussion channel, just return the zeros
        if self.is_drum:
            return synthesized
        # If the above if statement failed, we need to revert back to default
        if not hasattr(wave, '__call__'):
            raise ValueError('wave should be a callable Python function')
        # This is a simple way to make the end of the notes fade-out without
        # clicks
        fade_out = np.linspace(1, 0, .1*fs)
        # Create a frequency multiplier array for pitch bend
        bend_multiplier = np.ones(synthesized.shape)
        # Need to sort the pitch bend list for the loop below to work
        ordered_bends = sorted(self.pitch_bends, key=lambda bend: bend.time)
        # Add in a bend of 0 at the end of time
        end_bend = PitchBend(0, self.get_end_time())
        for start_bend, end_bend in zip(ordered_bends,
                                        ordered_bends[1:] + [end_bend]):
            # Bend start and end time in samples
            start = int(start_bend.time*fs)
            end = int(end_bend.time*fs)
            # The multiplier will be (twelfth root of 2)^(bend semitones)
            bend_semitones = pitch_bend_to_semitones(start_bend.pitch)
            bend_amount = (2**(1/12.))**bend_semitones
            # Sample indices effected by the bend
            bend_multiplier[start:end] = bend_amount
        # Add in waveform for each note
        for note in self.notes:
            # Indices in samples of this note
            start = int(fs*note.start)
            end = int(fs*note.end)
            # Get frequency of note from MIDI note number
            frequency = note_number_to_hz(note.pitch)
            # When a pitch bend gets applied, there will be a sample
            # discontinuity. So, we also need an array of offsets which get
            # applied to compensate.
            offsets = np.zeros(end - start)
            for bend in ordered_bends:
                bend_sample = int(bend.time*fs)
                # Does this pitch bend fall within this note?
                if bend_sample > start and bend_sample < end:
                    # Compute the average bend so far
                    bend_so_far = bend_multiplier[start:bend_sample].mean()
                    bend_amount = bend_multiplier[bend_sample]
                    # Compute the offset correction
                    offset = (bend_so_far - bend_amount)*(bend_sample - start)
                    # Store this offset for samples effected
                    offsets[bend_sample - start:] = offset
            # Compute the angular frequencies, bent, over this interval
            frequencies = 2*np.pi*frequency*(bend_multiplier[start:end])/fs
            # Synthesize using wave function at this frequency
            note_waveform = wave(frequencies*np.arange(end - start) +
                                 2*np.pi*frequency*offsets/fs)
            # Apply an exponential envelope
            envelope = np.exp(-np.arange(end - start)/(1.0*fs))
            # Make the end of the envelope be a fadeout
            if envelope.shape[0] > fade_out.shape[0]:
                envelope[-fade_out.shape[0]:] *= fade_out
            else:
                envelope *= np.linspace(1, 0, envelope.shape[0])
            # Multiply by velocity (don't think it's linearly scaled but
            # whatever)
            envelope *= note.velocity
            # Add in envelope'd waveform to the synthesized signal
            synthesized[start:end] += envelope*note_waveform

        return synthesized

    def fluidsynth(self, fs=44100, sf2_path=None):
        """Synthesize using fluidsynth.

        Parameters
        ----------
        fs : int
            Sampling rate to synthesize
        sf2_path : str
            Path to a .sf2 file.
            Default None, which uses the TimGM6mb.sf2 file included with
            pretty_midi.

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at fs

        """
        # If sf2_path is None, use the included TimGM6mb.sf2 path
        if sf2_path is None:
            sf2_path = pkg_resources.resource_filename(__name__, DEFAULT_SF2)

        if not _HAS_FLUIDSYNTH:
            raise ImportError("fluidsynth() was called but pyfluidsynth "
                              "is not installed.")

        if not os.path.exists(sf2_path):
            raise ValueError("No soundfont file found at the supplied path "
                             "{}".format(sf2_path))

        # Create fluidsynth instance
        fl = fluidsynth.Synth(samplerate=fs)
        # Load in the soundfont
        sfid = fl.sfload(sf2_path)
        # If this is a drum instrument, use channel 9 and bank 128
        if self.is_drum:
            channel = 9
            # Try to use the supplied program number
            res = fl.program_select(channel, sfid, 128, self.program)
            # If the result is -1, there's no preset with this program number
            if res == -1:
                # So use preset 0
                fl.program_select(channel, sfid, 128, 0)
        # Otherwise just use channel 0
        else:
            channel = 0
            fl.program_select(channel, sfid, 0, self.program)
        # Collect all notes in one list
        event_list = []
        for note in self.notes:
            event_list += [[note.start, 'note on', note.pitch, note.velocity]]
            event_list += [[note.end, 'note off', note.pitch]]
        for bend in self.pitch_bends:
            event_list += [[bend.time, 'pitch bend', bend.pitch]]
        for control_change in self.control_changes:
            event_list += [[control_change.time, 'control change',
                            control_change.number, control_change.value]]
        # Sort the event list by time, and secondarily by whether the event
        # is a note off
        event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
        # Add some silence at the beginning according to the time of the first
        # event
        current_time = event_list[0][0]
        # Convert absolute seconds to relative samples
        next_event_times = [e[0] for e in event_list[1:]]
        for event, end in zip(event_list[:-1], next_event_times):
            event[0] = end - event[0]
        # Include 1 second of silence at the end
        event_list[-1][0] = 1.
        # Pre-allocate output array
        total_time = current_time + np.sum([e[0] for e in event_list])
        synthesized = np.zeros(int(np.ceil(fs*total_time)))
        # Iterate over all events
        for event in event_list:
            # Process events based on type
            if event[1] == 'note on':
                fl.noteon(channel, event[2], event[3])
            elif event[1] == 'note off':
                fl.noteoff(channel, event[2])
            elif event[1] == 'pitch bend':
                fl.pitch_bend(channel, event[2])
            elif event[1] == 'control change':
                fl.cc(channel, event[2], event[3])
            # Add in these samples
            current_sample = int(fs*current_time)
            end = int(fs*(current_time + event[0]))
            samples = fl.get_samples(end - current_sample)[::2]
            synthesized[current_sample:end] += samples
            # Increment the current sample
            current_time += event[0]
        # Close fluidsynth
        fl.delete()

        return synthesized

    def __repr__(self):
        return 'Instrument(program={}, is_drum={})'.format(
            self.program, self.is_drum, len(self.notes))


class Note(object):
    """A note event.

    Parameters
    ----------
    velocity : int
        Note velocity
    pitch : int
        Note pitch, as a MIDI note number
    start : float
        Note on time, absolute, in seconds
    end : float
        Note off time, absolute, in seconds

    Notes
    -----
    This class is simply a container for its parameters.

    """

    def __init__(self, velocity, pitch, start, end):
        """Create a note object.

        """
        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end

    def __repr__(self):
        return 'Note(start={:f}, end={:f}, pitch={}, velocity={})'.format(
            self.start, self.end, self.pitch, self.velocity)


class PitchBend(object):
    """A pitch bend event.

    Parameters
    ----------
    pitch : int
        MIDI pitch bend amount, in the range [-8192, 8191]
    time : float
        Time where the pitch bend occurs

    Notes
    -----
    This class is simply a container for its parameters.

    """

    def __init__(self, pitch, time):
        """Create pitch bend object.

        """
        self.pitch = pitch
        self.time = time

    def __repr__(self):
        return 'PitchBend(pitch={:d}, time={:f})'.format(self.pitch, self.time)


class ControlChange(object):
    """A control change event.

    Parameters
    ----------
    number : int
        The control change number, in [0, 127]
    value : int
        The value of the control change, in [0, 127]
    time : float
        Time where the pitch bend occurs

    Notes
    -----
    This class is simply a container for its parameters.

    """

    def __init__(self, number, value, time):
        """Create control change object.

        """
        self.number = number
        self.value = value
        self.time = time

    def __repr__(self):
        return ('ControlChange(number={:d}, value={:d}, '
                'time={:f})'.format(self.number, self.value, self.time))


class TimeSignature(object):
    """Containts the time signature numerator, denominator and
    the event time in seconds

    Attributes
    ----------
        numerator : int
            Numerator of time signature
        denominator : int
            Denominator of time signature
        time : float
            Time of event in seconds

    Examples
    --------
    Instantiate a TimeSignature object with 6/8 time signature at 3.14 seconds
    >>> ts = TimeSignature(6, 8, 3.14)
    >>> print ts
    6/8 at 3.14 seconds

    """

    def __init__(self, numerator, denominator, time):
        assert isinstance(numerator, (int, np.int)), \
        '%s is not a recognized `numerator` type' % str(type(numerator))
        assert isinstance(denominator, (int, np.int)), \
        '%s is not a recognized `denominator` type' % str(type(denominator))
        assert isinstance(time, (float, np.float)), \
        '%s is not a recognized `time` type' % str(type(key_number))

        self.numerator = numerator
        self.denominator = denominator
        self.time = time

    def __repr__(self):
        return '%d/%d at %.2f seconds' % (
            self.numerator,
            self.denominator,
            self.time)


class KeySignature(object):
    """Contains the key signature and the event time in seconds

    Only supports major and minor keys.

    Attributes
    ----------
        key_number : int
            key number accordingly to [0,11] Major, [12,23] minor
            For example, 0 is C Major, 12 is C minor
        time : float
            time of event in seconds
    Examples
    --------
    Instantiate a C# minor KeySignature object at 3.14 seconds.
    >>> ks = KeySignature(13, 3.14)
    >>> print ks
    C# minor at 3.14 seconds
    """

    def __init__(self, key_number, time):
        assert isinstance(key_number, (int, np.int)), \
        '%s is not a recognized `key_number` type' % str(type(key_number))
        assert isinstance(time, (float, np.float)), \
        '%s is not a recognized `time` type' % str(type(key_number))

        self.key_number = key_number
        self.time = time

    def __repr__(self):
        return '%s at %.2f seconds' % (
            KeySignature.key_number_to_key_string(self.key_number),
            self.time)

    @staticmethod
    def key_number_to_key_string(key_number):
        """Convert a key number to a key string

        Parameters
        ----------
            key_number : int
                Uses pitch classes to represent major keys.
                For minor keys, adds a 12 offset.
                For example, C major is 0 and C minor is 12.

        Returns
        -------
        str
            Root and mode, e.g. Gb minor.
            Gives preference for keys with flats, with the
            exception of F#, G# and C# minor.
        """

        assert isinstance(key_number, (int, np.int)), 'key_number is not int!'
        assert ((key_number >= 0) and (key_number < 24)), 'key_number is'
        'larger than 24'

        # preference to keys with flats
        keys = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb',
                'G', 'Ab', 'A', 'Bb', 'B']
        key_idx = key_number % 12
        mode = key_number / 12

        if mode == 0:
            return keys[key_idx] + ' Major'
        elif mode == 1:
            # preference to C#, F# and G# minor
            if key_idx in [1, 6, 8]:
                return keys[key_idx-1] + '# minor'
            else:
                return keys[key_idx] + ' minor'

    @staticmethod
    def key_string_to_key_number(key_string):
        """Convert a correctly formated string key to key number

        Parameters
            key_string : str
                Key as string. format is 'key mode', where:
                    key is notaded using ABCDEFG with # or b;
                    mode is notated using 'major' or 'minor'.
                Letter case is irrelevant for key and mode.
        """

        assert isinstance(key_string, str), "KeyString is not String"
        assert key_string[1] in ['#', 'b', ' '], "Second character"
        " %s is not #, b nor blank_space" % key_string[1]

        # split key and mode, ignore case
        key_str, mode_str = key_string.split()
        key_str = key_str.upper()
        mode_str = mode_str.lower()

        # instantiate default pitch classes and supported modes
        note_names_pc = {'C': 0, 'D': 2, 'E': 4,
                         'F': 5, 'G': 7, 'A': 9, 'B': 11}
        modes = ['major', 'minor']

        # check that both key and mode are valid
        assert key_str[0] in note_names_pc, 'Key is not recognized'
        assert mode_str in modes, 'Mode is not recognized'

        # lookup dictionary
        key_number = note_names_pc[key_str[0]]

        # offset key_index according to sharp or flat key
        key_offset = 0
        if len(key_str) == 2:
            if key_str[1] == '#':
                key_number += 1
            else:
                key_number -= 1

        # circle around 12
        key_number = key_number % 12

        # offset if mode is minor
        if mode_str == 'minor':
            key_number += 12

        return key_number


def note_number_to_hz(note_number):
    """Convert a (fractional) MIDI note number to its frequency in Hz.

    Parameters
    ----------
    note_number : float
        MIDI note number, can be fractional

    Returns
    -------
    note_frequency : float
        Frequency of the note in Hz

    """
    # MIDI note numbers are defined as the number of semitones relative to C0
    # in a 440 Hz tuning
    return 440.0*(2.0**((note_number - 69)/12.0))


def hz_to_note_number(frequency):
    """Convert a frequency in Hz to a (fractional) frequency

    Parameters
    ----------
    frequency : float
        Frequency of the note in Hz

    Returns
    -------
    note_number : float
        MIDI note number, can be fractional

    """
    # MIDI note numbers are defined as the number of semitones relative to C0
    # in a 440 Hz tuning
    return 12*(np.log2(frequency) - np.log2(440.0)) + 69


def note_name_to_number(note_name):
    """Converts a note name in the format (note)(accidental)(octave number)
    to MIDI note number.

    (note) is required, and is case-insensitive.

    (accidental) should be '' for natural, '#' for sharp and '!' or 'b' for
    flat.

    If (octave) is '', octave 0 is assumed.

    Parameters
    ----------
    note_name : str
        A note name, as described above

    Returns
    -------
    note_number : int
        MIDI note number corresponding to the provided note name.

    Notes
    -----
        Thanks to Brian McFee.

    """

    # Map note name to the semitone
    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    # Relative change in semitone denoted by each accidental
    acc_map = {'#': 1, '': 0, 'b': -1, '!': -1}

    # Reg exp will raise an error when the note name is not valid
    try:
        # Extract pitch, octave, and accidental from the supplied note name
        match = re.match(r'^(?P<n>[A-Ga-g])(?P<off>[#b!]?)(?P<oct>[+-]?\d+)$',
                         note_name)

        pitch = match.group('n').upper()
        offset = acc_map[match.group('off')]
        octave = int(match.group('oct'))
    except:
        raise ValueError('Improper note format: %s' % note_name)

    # Convert from the extrated ints to a full note number
    return 12*octave + pitch_map[pitch] + offset


def note_number_to_name(note_number):
    """Convert a MIDI note number to its name, in the format
    (note)(accidental)(octave number) (e.g. 'C#4')

    Parameters
    ----------
    note_number : int
        MIDI note number.  If not an int, it will be rounded.

    Returns
    -------
    note_name : str
        Name of the supplied MIDI note number.

    Notes
    -----
        Thanks to Brian McFee.

    """

    # Note names within one octave
    semis = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Ensure the note is an int
    note_number = int(np.round(note_number))

    # Get the semitone and the octave, and concatenate to create the name
    return semis[note_number % 12] + str(note_number/12)

# List which maps MIDI note number - 35 to drum name
# from http://www.midi.org/techspecs/gm1sound.php
__DRUM_MAP = ['Acoustic Bass Drum', 'Bass Drum 1', 'Side Stick',
              'Acoustic Snare', 'Hand Clap', 'Electric Snare',
              'Low Floor Tom', 'Closed Hi Hat', 'High Floor Tom',
              'Pedal Hi Hat', 'Low Tom', 'Open Hi Hat',
              'Low-Mid Tom', 'Hi-Mid Tom', 'Crash Cymbal 1',
              'High Tom', 'Ride Cymbal 1', 'Chinese Cymbal',
              'Ride Bell', 'Tambourine', 'Splash Cymbal',
              'Cowbell', 'Crash Cymbal 2', 'Vibraslap',
              'Ride Cymbal 2', 'Hi Bongo', 'Low Bongo',
              'Mute Hi Conga', 'Open Hi Conga', 'Low Conga',
              'High Timbale', 'Low Timbale', 'High Agogo',
              'Low Agogo', 'Cabasa', 'Maracas',
              'Short Whistle', 'Long Whistle', 'Short Guiro',
              'Long Guiro', 'Claves', 'Hi Wood Block',
              'Low Wood Block', 'Mute Cuica', 'Open Cuica',
              'Mute Triangle', 'Open Triangle']


def note_number_to_drum_name(note_number):
    """Converts a MIDI note number in a percussion instrument to the
    corresponding drum name, according to the General MIDI standard.

    Any MIDI note number outside of the valid range (note 35-81, zero-indexed)
    will result in an empty string.

    Parameters
    ----------
    note_number : int
        MIDI note number.  If not an int, it will be rounded.

    Returns
    -------
    drum_name : str
        Name of the drum for this note for a percussion instrument.

    Notes
    -----
        See http://www.midi.org/techspecs/gm1sound.php

    """

    # Ensure note is an int
    note_number = int(np.round(note_number))
    # General MIDI only defines drum names for notes 35-81
    if note_number < 35 or note_number > 81:
        return ''
    else:
        # Our __DRUM_MAP starts from index 0; drum names start from 35
        return __DRUM_MAP[note_number - 35]


def __normalize_str(name):
    """Removes all non-alphanumeric characters from a string and converts
    it to lowercase.

    """
    return ''.join(ch for ch in name if ch.isalnum()).lower()


def drum_name_to_note_number(drum_name):
    """Converts a drum name to the corresponding MIDI note number for a
    percussion instrument.  Conversion is case, whitespace, and
    non-alphanumeric character insensitive.

    Parameters
    ----------
    drum_name : str
        Name of a drum which exists in the general MIDI standard.
        If the drum is not found, a ValueError is raised.

    Returns
    -------
    note_number : int
        The MIDI note number corresponding to this drum.

    Notes
    -----
        See http://www.midi.org/techspecs/gm1sound.php

    """

    normalized_drum_name = __normalize_str(drum_name)
    # Create a list of the entries __DRUM_MAP, normalized, to search over
    normalized_drum_names = [__normalize_str(name) for name in __DRUM_MAP]

    # If the normalized drum name is not found, complain
    try:
        note_index = normalized_drum_names.index(normalized_drum_name)
    except:
        raise ValueError('{} is not a valid General MIDI drum '
                         'name.'.format(drum_name))

    # If an index was found, it will be 0-based; add 35 to get the note number
    return note_index + 35


__INSTRUMENT_MAP = ['Acoustic Grand Piano', 'Bright Acoustic Piano',
                    'Electric Grand Piano', 'Honky-tonk Piano',
                    'Electric Piano 1', 'Electric Piano 2', 'Harpsichord',
                    'Clavinet', 'Celesta', 'Glockenspiel', 'Music Box',
                    'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells',
                    'Dulcimer', 'Drawbar Organ', 'Percussive Organ',
                    'Rock Organ', 'Church Organ', 'Reed Organ', 'Accordion',
                    'Harmonica', 'Tango Accordion', 'Acoustic Guitar (nylon)',
                    'Acoustic Guitar (steel)', 'Electric Guitar (jazz)',
                    'Electric Guitar (clean)', 'Electric Guitar (muted)',
                    'Overdriven Guitar', 'Distortion Guitar',
                    'Guitar Harmonics', 'Acoustic Bass',
                    'Electric Bass (finger)', 'Electric Bass (pick)',
                    'Fretless Bass', 'Slap Bass 1', 'Slap Bass 2',
                    'Synth Bass 1', 'Synth Bass 2', 'Violin', 'Viola', 'Cello',
                    'Contrabass', 'Tremolo Strings', 'Pizzicato Strings',
                    'Orchestral Harp', 'Timpani', 'String Ensemble 1',
                    'String Ensemble 2', 'Synth Strings 1', 'Synth Strings 2',
                    'Choir Aahs', 'Voice Oohs', 'Synth Choir', 'Orchestra Hit',
                    'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet',
                    'French Horn', 'Brass Section', 'Synth Brass 1',
                    'Synth Brass 2', 'Soprano Sax', 'Alto Sax', 'Tenor Sax',
                    'Baritone Sax', 'Oboe', 'English Horn', 'Bassoon',
                    'Clarinet', 'Piccolo', 'Flute', 'Recorder', 'Pan Flute',
                    'Blown bottle', 'Shakuhachi', 'Whistle', 'Ocarina',
                    'Lead 1 (square)', 'Lead 2 (sawtooth)',
                    'Lead 3 (calliope)', 'Lead 4 chiff', 'Lead 5 (charang)',
                    'Lead 6 (voice)', 'Lead 7 (fifths)',
                    'Lead 8 (bass + lead)', 'Pad 1 (new age)', 'Pad 2 (warm)',
                    'Pad 3 (polysynth)', 'Pad 4 (choir)', 'Pad 5 (bowed)',
                    'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)',
                    'FX 1 (rain)', 'FX 2 (soundtrack)', 'FX 3 (crystal)',
                    'FX 4 (atmosphere)', 'FX 5 (brightness)', 'FX 6 (goblins)',
                    'FX 7 (echoes)', 'FX 8 (sci-fi)', 'Sitar', 'Banjo',
                    'Shamisen', 'Koto', 'Kalimba', 'Bagpipe', 'Fiddle',
                    'Shanai', 'Tinkle Bell', 'Agogo', 'Steel Drums',
                    'Woodblock', 'Taiko Drum', 'Melodic Tom', 'Synth Drum',
                    'Reverse Cymbal', 'Guitar Fret Noise', 'Breath Noise',
                    'Seashore', 'Bird Tweet', 'Telephone Ring', 'Helicopter',
                    'Applause', 'Gunshot']


def program_to_instrument_name(program_number):
    """Converts a MIDI program number to the corresponding General MIDI
    instrument name.

    Parameters
    ----------
    program_number : int
        MIDI program number, between 0 and 127

    Returns
    -------
    instrument_name : str
        Name of the instrument corresponding to this program number.

    Notes
    -----
        See http://www.midi.org/techspecs/gm1sound.php

    """

    # Check that the supplied program is in the valid range
    if program_number < 0 or program_number > 127:
        raise ValueError('Invalid program number {}, should be between 0 and'
                         ' 127'.format(program_number))
    # Just grab the name from the instrument mapping list
    return __INSTRUMENT_MAP[program_number]


def instrument_name_to_program(instrument_name):
    """Converts an instrument name to the corresponding General MIDI program
    number.  Conversion is case, whitespace, and non-alphanumeric character
    insensitive.

    Parameters
    ----------
    instrument_name : str
        Name of an instrument which exists in the general MIDI standard.
        If the instrument is not found, a ValueError is raised.

    Returns
    -------
    program_number : int
        The MIDI program number corresponding to this instrument.

    Notes
    -----
        See http://www.midi.org/techspecs/gm1sound.php

    """

    normalized_inst_name = __normalize_str(instrument_name)
    # Create a list of the entries __INSTRUMENT_MAP, normalized, to search over
    normalized_inst_names = [__normalize_str(name) for name in
                             __INSTRUMENT_MAP]

    # If the normalized drum name is not found, complain
    try:
        program_number = normalized_inst_names.index(normalized_inst_name)
    except:
        raise ValueError('{} is not a valid General MIDI instrument '
                         'name.'.format(instrument_name))

    # Return the index (program number) if a match was found
    return program_number


__INSTRUMENT_CLASSES = ['Piano', 'Chromatic Percussion', 'Organ', 'Guitar',
                        'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe',
                        'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic',
                        'Percussive',
                        'Sound Effects']


def program_to_instrument_class(program_number):
    """Converts a MIDI program number to the corresponding General MIDI
    instrument class.

    Parameters
    ----------
    program_number : int
        MIDI program number, between 0 and 127

    Returns
    -------
    instrument_class : str
        Name of the instrument class corresponding to this program number.

    Notes
    -----
        See http://www.midi.org/techspecs/gm1sound.php

    """

    # Check that the supplied program is in the valid range
    if program_number < 0 or program_number > 127:
        raise ValueError('Invalid program number {}, should be between 0 and'
                         ' 127'.format(program_number))
    # Just grab the name from the instrument mapping list
    return __INSTRUMENT_CLASSES[int(program_number)/8]


def pitch_bend_to_semitones(pitch_bend, semitone_range=2.):
    """Convert a MIDI pitch bend value (in the range -8192, 8191) to the bend
    amount in semitones.

    Parameters
    ----------
    pitch_bend : int
        MIDI pitch bend amount, in [-8192, 8191]
    semitone_range : float
        Convert to +/- this semitone range.  Default is 2., which is the
        General MIDI standard +/-2 semitone range.

    Returns
    -------
    semitones : float
        Number of semitones corresponding to this pitch bend amount

    """

    return semitone_range*pitch_bend/8192.0


def semitones_to_pitch_bend(semitones, semitone_range=2.):
    """Convert a semitone value to the corresponding MIDI pitch bend int

    Parameters
    ----------
    semitones : float
        Number of semitones for the pitch bend
    semitone_range : float
        Convert to +/- this semitone range.  Default is 2., which is the
        General MIDI standard +/-2 semitone range.

    Returns
    -------
    pitch_bend : int
        MIDI pitch bend amount, in [-8192, 8191]

    """
    return int(8192*(semitones/semitone_range))


def midi_key_to_key_name(key_signature_event):
    """Convert midi package's midi.event.KeySignature to pretty_midi's key_number

    Parameter
    ---------
        key_signature_event : midi.event.KeySignature
            Converts the midi.event.KeySignature to conform with
            pretty_midi's key_number.
    """

    sharp_keys = 'CGDAEBF'
    flat_keys = 'CFBEADG'
    num_accidentals, mode = key_signature_event.data

    # check if key signature has sharps or flats
    if num_accidentals >= 0 and num_accidentals < 2**7:
        num_sharps = num_accidentals / 6
        key = sharp_keys[num_accidentals % 7] + '#' * num_sharps
    else:
        num_accidentals = 256 - num_accidentals
        num_flats = num_accidentals / 2
        key = flat_keys[num_accidentals % 7] + 'b' * num_flats

    # append mode to string
    if mode == 0:
        key += ' Major'
    else:
        key += ' minor'

    # use routine to convert from stringt notation to number notation
    return KeySignature.key_string_to_key_number(key)
