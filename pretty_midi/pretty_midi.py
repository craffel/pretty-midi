"""Utility functions for handling MIDI data in an easy to read/manipulate
format

"""

import midi
import numpy as np
import warnings
import collections
import copy

from .instrument import Instrument
from .containers import KeySignature, TimeSignature
from .containers import Note, PitchBend, ControlChange
from .utilities import mode_accidentals_to_key_number
from .utilities import key_number_to_mode_accidentals
from .utilities import qpm_to_bpm

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

            # Populate the list of tempo changes (tick scales)
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

            # Check that there are tempo, key and time change events
            # only on track 0
            if sum([sum([isinstance(event, (midi.events.SetTempoEvent,
                                            midi.events.KeySignatureEvent,
                                            midi.events.TimeSignatureEvent))
                        for event in track]) for track in midi_data[1:]]):
                warnings.warn(("Tempo, Key or Time signature change events"
                               " found on non-zero tracks."
                               "  This is not a valid type 0 or type 1 MIDI"
                               " file. Tempo, Key or Time Signature"
                               " may be wrong."),
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
            # Empty key signature changes list
            self.key_signature_changes = []
            # Empty time signatures changes list
            self.time_signature_changes = []

    def _load_tempo_changes(self, midi_data):
        """Populates self.__tick_scales with tuples of (tick, tick_scale)

        Parameters
        ----------
        midi_data : midi.FileReader
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
        """Populates self.time_signature_changes with TimeSignature objects and
        populates self.key_signature_changes with KeySignature objects.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read
        """

        # list to store key signature changes
        self.key_signature_changes = []

        # list to store time signatures changes
        self.time_signature_changes = []

        for event in midi_data[0]:
            if isinstance(event, midi.events.KeySignatureEvent):
                key_obj = KeySignature(mode_accidentals_to_key_number(
                    event.data[1], event.get_alternatives()),
                    self.__tick_to_time[event.tick])
                self.key_signature_changes.append(key_obj)

            elif isinstance(event, midi.events.TimeSignatureEvent):
                ts_obj = TimeSignature(event.get_numerator(),
                                       event.get_denominator(),
                                       self.__tick_to_time[event.tick])
                self.time_signature_changes.append(ts_obj)

    def _update_tick_to_time(self, max_tick):
        """Creates __tick_to_time, a class member array which maps ticks to
        time starting from tick 0 and ending at max_tick

        Parameters
        ----------
        max_tick : int
            last tick to compute time for

        """
        # Allocate tick to time array - indexed by tick from 0 to max_tick
        self.__tick_to_time = np.zeros(max_tick + 1)
        # Keep track of the end time of the last tick in the previous interval
        last_end_time = 0
        # Cycle through intervals of different tempi
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
        # MIDI files can contain a collection of tracks; each track can have
        # events occuring on one of sixteen channels, and events can correspond
        # to different instruments according to the most recently occurring
        # program number.  So, we need a way to keep track of which instrument
        # is playing on each track on each channel.  This dict will map from
        # program number, drum/not drum, channel, and track index to instrument
        # indices, which we will retrieve/populate using the __get_instrument
        # function below.
        instrument_map = {}

        def __get_instrument(program, is_drum, channel, track):
            """Gets the Instrument corresponding to the given program number,
            drum/non-drum type, channel, and track index.  If no such
            instrument exists, one is created.

            """
            # If we have already created an instrument for this program
            # number/track/channel, return it
            if (program, is_drum, channel, track) in instrument_map:
                return instrument_map[(program, is_drum, channel, track)]
            # Create the instrument if none was found
            self.instruments.append(Instrument(program, is_drum))
            instrument = self.instruments[-1]
            # Add the instrument to the instrument map
            instrument_map[(program, is_drum, channel, track)] = instrument
            return instrument

        # Initialize empty list of instruments
        self.instruments = []
        for track_idx, track in enumerate(midi_data):
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
                elif event.name == 'Note Off' or (event.name == 'Note On' and
                                                  event.velocity == 0):
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
                            instrument = __get_instrument(
                                program, is_drum, event.channel, track_idx)
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
                    instrument = __get_instrument(
                        program, is_drum, event.channel, track_idx)
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
                    instrument = __get_instrument(
                        program, is_drum, event.channel, track_idx)
                    # Add the control change event
                    instrument.control_changes.append(control_change)

    def get_tempo_changes(self):
        """Return arrays of tempo changes and their times.

        This is direct from the MIDI file.

        Returns
        -------
        tempo_change_times : np.ndarray
            Times, in seconds, where the tempo changes.
        tempi : np.ndarray
            What the tempo is at each point in time in tempo_change_times

        """

        # Pre-allocate return arrays
        tempo_change_times = np.zeros(len(self.__tick_scales))
        tempi = np.zeros(len(self.__tick_scales))
        for n, (tick, tick_scale) in enumerate(self.__tick_scales):
            # Convert tick of this tempo change to time in seconds
            tempo_change_times[n] = self.__tick_to_time[tick]
            # Convert tick scale to a tempo
            tempi[n] = 60.0/(tick_scale*self.resolution)
        return tempo_change_times, tempi

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

    def estimate_tempi(self):
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
                clusters[k] = (cluster_counts[k]*clusters[k] +
                               interval)/(cluster_counts[k] + 1)
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
        """Returns the best tempo estimate from estimate_tempi(), for
        convenience

        Returns
        -------
        tempo : float
            Estimated tempo, in bpm

        """
        return self.estimate_tempi()[0][0]

    def get_beats(self, start_time=0.):
        """Return a list of beat locations, according to MIDI tempo changes.
        Will not be correct if the MIDI data has been modified without changing
        tempo information.

        Parameters
        ----------
        start_time : float
            Location of the first beat, in seconds.

        Returns
        -------
        beats : np.ndarray
            Beat locations, in seconds.

        """
        # Get tempo changes and tempos
        tempo_change_times, tempi = self.get_tempo_changes()
        # Create beat list; first beat is at first onset
        beats = [start_time]
        # Index of the tempo we're using
        tempo_idx = 0
        # Move past all the tempo changes up to the supplied start time
        while (tempo_idx < tempo_change_times.shape[0] - 1 and
                beats[-1] > tempo_change_times[tempo_idx]):
            tempo_idx += 1
        # Index of the time signature change we're using
        ts_idx = 0
        # Move past all time signature changes up to the supplied start time
        while (ts_idx < len(self.time_signature_changes) - 1 and
                beats[-1] > self.time_signature_changes[ts_idx]):
            ts_idx += 1

        def get_current_bpm():
            ''' Convenience function which computs the current BPM based on the
            current tempo change and time signature events '''
            # When there are time signature changes, use them to compute BPM
            if len(self.time_signature_changes) > 0:
                return qpm_to_bpm(
                    tempi[tempo_idx],
                    self.time_signature_changes[ts_idx].numerator,
                    self.time_signature_changes[ts_idx].denominator)
            # Otherwise, just use the raw tempo change event tempo
            else:
                return tempi[tempo_idx]
        # Get track end time
        end_time = self.get_end_time()
        # Add beats in
        while beats[-1] < end_time:
            # Update the current bpm
            bpm = get_current_bpm()
            # Compute expected beat location, one period later
            next_beat = beats[-1] + 60.0/bpm
            # If the next beat location passes a time signature change boundary
            if ts_idx < len(self.time_signature_changes) - 1:
                # Time of the next time signature change
                next_ts_time = self.time_signature_changes[ts_idx + 1].time
                if (next_beat > next_ts_time or
                        np.isclose(next_beat, next_ts_time)):
                    # Set the next beat to the time signature change time
                    next_beat = self.time_signature_changes[ts_idx + 1].time
                    # Update the time signature index
                    ts_idx += 1
                    # Update the current bpm
                    bpm = get_current_bpm()
            # If the beat location passes a tempo change boundary...
            if (tempo_idx < tempo_change_times.shape[0] - 1 and
                    next_beat > tempo_change_times[tempo_idx + 1]):
                # Start by setting the beat location to the current beat...
                next_beat = beats[-1]
                # with the entire beat remaining
                beat_remaining = 1.0
                # While a beat with the current tempo would pass a tempo
                # change boundary...
                while (tempo_idx < tempo_change_times.shape[0] - 1 and
                        next_beat + beat_remaining*60.0/bpm >=
                        tempo_change_times[tempo_idx + 1]):
                    # Update the current bpm
                    bpm = get_current_bpm()
                    # Compute the amount the beat location overshoots
                    overshot_ratio = (tempo_change_times[tempo_idx + 1] -
                                      next_beat)/(60.0/bpm)
                    # Add in the amount of the beat during this tempo
                    next_beat += overshot_ratio*60.0/bpm
                    # Less of the beat remains now
                    beat_remaining -= overshot_ratio
                    # Increment the tempo index
                    tempo_idx = tempo_idx + 1
                # Update the current bpm
                bpm = get_current_bpm()
                next_beat += beat_remaining*60./bpm
            beats.append(next_beat)
        # The last beat will pass the end_time barrier, so don't include it
        beats = np.array(beats[:-1])
        return beats

    def estimate_beat_start(self, candidates=10, tolerance=.025):
        """Estimate the location of the first beat based on which of the first
        few onsets results in the best correlation with the onset spike train.

        Parameters
        ----------
        candidates : int
            Number of candidate onsets to try
        tolerance : float
            The tolerance in seconds around which onsets will be used to
            treat a beat as correct

        Returns
        -------
        beat_start : float
            The offset which is chosen as the beat start location
        """
        # Get a sorted list of all notes from all instruments
        note_list = [n for i in self.instruments for n in i.notes]
        note_list.sort(key=lambda note: note.start)
        # List of possible beat trackings
        beat_candidates = []
        # List of start times for each beat candidate
        start_times = []
        onset_index = 0
        # Try the first 10 (unique) onsets as beat tracking start locations
        while (len(beat_candidates) <= candidates and
               len(beat_candidates) <= len(note_list) and
               onset_index < len(note_list)):
            # Make sure we are using a new start location
            if onset_index == 0 or np.abs(note_list[onset_index - 1].start -
                                          note_list[onset_index].start) > .001:
                beat_candidates.append(
                    self.get_beats(note_list[onset_index].start))
                start_times.append(note_list[onset_index].start)
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
                if beat - tolerance < 0:
                    beat_window = np.ones(
                        int(fs*2*tolerance + (beat - tolerance)*fs))
                    beat_signal[:int((beat + tolerance)*fs)] = beat_window
                else:
                    beat_start = int((beat - tolerance)*fs)
                    beat_end = beat_start + int(fs*tolerance*2)
                    beat_window = np.ones(int(fs*tolerance*2))
                    beat_signal[beat_start:beat_end] = beat_window
            # Compute their dot product and normalize to get score
            onset_scores[n] = np.dot(beat_signal, onset_signal)/beats.shape[0]
        # Return the best-scoring beat start
        return start_times[np.argmax(onset_scores)]

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

    def get_piano_roll(self, fs=100, times=None):
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
        piano_rolls = [i.get_piano_roll(fs=fs, times=times)
                       for i in self.instruments]
        # Allocate piano roll,
        # number of columns is max of # of columns in all piano rolls
        piano_roll = np.zeros((128, np.max([p.shape[1] for p in piano_rolls])))
        # Sum each piano roll into the aggregate piano roll
        for roll in piano_rolls:
            piano_roll[:, :roll.shape[1]] += roll
        return piano_roll

    def get_pitch_class_histogram(self, use_duration=False,
                                  use_velocity=False, normalize=True):
        """Computes the histogram of pitch classes given all tracks

        Parameters
        ----------
        use_duration : bool
            Weight frequency by note duration
        use_velocity : bool
            Weight frequency by note velocity
        normalize : bool
            Normalizes the histogram such that the sum of bin values is 1.

        Returns
        -------
        histogram : np.ndarray, shape=(12,)
            Histogram of pitch classes given all tracks, optionally weighted
            by their durations or velocities
        """
        # Sum up all histograms from all instruments defaulting to np.zeros(12)
        histogram = sum([
            i.get_pitch_class_histogram(use_duration, use_velocity)
            for i in self.instruments], np.zeros(12))

        # Normalize accordingly
        if normalize:
            histogram /= (histogram.sum() + (histogram.sum() == 0))

        return histogram

    def get_pitch_class_transition_matrix(self, normalize=False,
                                          time_thresh=0.05):
        """Computes the total pitch class transition matrix of all instruments

        Transitions are added whenever the end of a note is within time_tresh
        from the start of any other note.

        Parameters
        ----------
        normalize : bool
            Normalize transition matrix such that matrix sum equals is 1.
        time_thresh : float
            Maximum temporal threshold, in seconds, between the start of a note
            and end time of any other note for a transition to be added.

        Returns
        -------
        pitch_class_transition_matrix : np.ndarray, shape=(12,12)
            Pitch class transition matrix given all tracks
        """
        # Sum up all matrices from all instruments defaulting zeros matrix
        pc_trans_mat = sum(
            [i.get_pitch_class_transition_matrix(normalize, time_thresh)
             for i in self.instruments], np.zeros((12, 12)))

        # Normalize accordingly
        if normalize:
            pc_trans_mat /= (pc_trans_mat.sum() + (pc_trans_mat.sum() == 0))

        return pc_trans_mat

    def get_chroma(self, fs=100, times=None):
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
        piano_roll = self.get_piano_roll(fs=fs, times=times)
        # Fold into one octave
        chroma_matrix = np.zeros((12, piano_roll.shape[1]))
        for note in range(12):
            chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)
        return chroma_matrix

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
        # If there are no instruments, or all instruments have no notes, return
        # an empty array
        if len(self.instruments) == 0 or all(len(i.notes) == 0
                                             for i in self.instruments):
            return np.array([])
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
        # Add in each time signature
        for ts in self.time_signature_changes:
            midi_ts = midi.events.TimeSignatureEvent()
            midi_ts.set_numerator(ts.numerator)
            midi_ts.set_denominator(ts.denominator)
            midi_ts.tick = self.time_to_tick(ts.time)
            timing_track += [midi_ts]
        # Add in each key signature
        for ks in self.key_signature_changes:
            midi_ks = midi.events.KeySignatureEvent()
            mode, num_accidentals = key_number_to_mode_accidentals(
                ks.key_number)
            midi_ks.set_alternatives(num_accidentals)
            midi_ks.set_minor(mode)
            midi_ks.tick = self.time_to_tick(ks.time)
            timing_track += [midi_ks]
        # Sort the (absolute-tick-timed) events.
        timing_track.sort(key=lambda event: event.tick)
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
