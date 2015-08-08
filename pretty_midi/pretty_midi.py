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
from .utilities import key_name_to_key_number

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
        """Populates self.time_signature_changes with TimeSignature objects and
        populates self.key_changes with KeySignature objects.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read
        """

        # helper function to get key number from midi key
        def midi_key_to_key_number(key_signature_event):
            """Convert midi package's midi.event.KeySignature to pretty_midi's
            key_number

            Parameters
            ----------
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

            # use routine to convert from string notation to number notation
            return key_name_to_key_number(key)

        # _load_metadata routine proper starts here
        # list to store key signature changes
        self.key_changes = []

        # list to store time signatures changes
        self.time_signature_changes = []

        for event in midi_data[0]:
            if isinstance(event, midi.events.KeySignatureEvent):
                key_obj = KeySignature(midi_key_to_key_number(event),
                                       self.__tick_to_time[event.tick])
                self.key_changes.append(key_obj)

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
            if (instrument.program == program and
                    instrument.is_drum == is_drum):
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

    def get_beats_using_metadata(self):
        """Uses Time Signature and Tempo metadata to estimate beat times

        Returns
        -------
            np.ndarray of floats, shape(#beats,)
                List of beat times in seconds
        """

        # Get tempo changes and tempi based on quarter note
        tempo_change_times, tempi = self.get_tempo_changes()

        # if there's only one tempo
        if len(tempo_change_times) == 1:
            # if there's only one time signature
            if len(self.time_signature_changes) == 1:
                # get tempo given time signature
                tempo = qpm_to_bpm(tempi[0], self.time_signature_changes[0])
                # interpolate through the end with given tempo
                timestamps = np.arange(tempo_change_times[0], self.get_end_time(), 60.0/tempo)
                return timestamps
            # if there is more than one time signature, update tempo accordingly
            else:
                start_time = 0
                timestamps = None
                for i in xrange(1, len(self.time_signature_changes)):
                    cur_ts = self.time_signature_changes[i-1]
                    nxt_ts = self.time_signature_changes[i]
                    # convert qpm to bpm
                    tempo = qpm_to_bpm(tempi[0], self.time_signature_changes[i-1])
                    if timestamps is None:
                        timestamps = np.arange(cur_ts.time, nxt_ts.time, 60.0/tempo)
                    else:
                        timestamps = np.hstack(timestamps, np.arange(cur_ts.time, nxt_ts.time, 60.0/tempo))
                # last time signature
                tempo = qpm_to_bpm(tempi[0], self.time_signature_changes[-1])
                timestamps = np.hstack(timestamps, np.arange(nxt_ts.time, self.get_end_time(), 60.0/tempo))
                return timestamps
        # if there are multiple tempi
        else:
            # if there's only one time signature
            if len(self.time_signature_changes) == 1:
                cur_beat = 0
                time_data_matrix = []

                #extract beat locations given tempi and their location in time
                for i in xrange(1, len(tempo_change_times)):
                    tempo = qpm_to_bpm(tempi[i-1], self.time_signature_changes[0])
                    cur_tempo_change_time = tempo_change_times[i-1]
                    nxt_tempo_change_time = tempo_change_times[i]

                    # iterate through beats
                    beat_len = 60.0 / tempo
                    beat_dur = (nxt_tempo_change_time - cur_tempo_change_time) / beat_len
                    time_data_matrix.append((cur_tempo_change_time, beat_len, cur_beat))
                    cur_beat += Fraction(beat_dur).limit_denominator(16)

                #convert to np.ndarray for convenience
                time_data_matrix = np.array(time_data_matrix)

                #given beat, find timestamp in seconds
                timestamps = []
                last_beat = time_data_matrix[:,2][-1]

                beats = np.arange(1, last_beat, beat_resolution)

                for beat in beats:
                    cur_idx = np.argmax(time_data_matrix[:,2] > beat) - 1
                    cur_time = time_data_matrix[cur_idx, 0]
                    cur_beat_len = time_data_matrix[cur_idx, 1]
                    cur_beat = time_data_matrix[cur_idx, 2]

                    if cur_beat == beat:
                        timestamps.append(cur_time)
                    else:
                        beat_dif = beat - cur_beat
                        beat_time = cur_time + cur_beat_len * beat_dif
                        timestamps.append(beat_time)
                return np.array(timestamps)
            # if there are multiple tempi and time signatures
            else:
                cur_beat = 0
                time_data_matrix = []

                # store TimeSignature objects
                ts_idx = 1
                cur_ts = self.time_signature_changes[ts_idx-1]
                nxt_ts = self.time_signature_changes[ts_idx]

                # extract beat locations given tempi and their location in time
                for i in xrange(1, len(tempo_change_times)):
                    cur_tempo_change_time = tempo_change_times[i-1]
                    nxt_tempo_change_time = tempo_change_times[i]

                    while nxt_ts < nxt_tempo_change_time:
                        tempo = qpm_to_bpm(tempi[i-1], cur_ts)

                        # iterate through beats
                        beat_len = 60.0 / tempo
                        beat_dur = (nxt_ts.time - cur_ts.time) / beat_len
                        time_data_matrix.append((cur_ts.time, beat_len, cur_beat))
                        cur_beat += Fraction(beat_dur).limit_denominator(16)

                        # update TimeSignature
                        ts_idx += 1
                        cur_ts = self.time_signature_changes[ts_idx-1]
                        nxt_ts = self.time_signature_changes[ts_idx]

                # convert to np.ndarray for convenience
                time_data_matrix = np.array(time_data_matrix)

                # given beat, find timestamp in seconds
                timestamps = []
                last_beat = time_data_matrix[:,2][-1]

                beats = np.arange(1, last_beat, beat_resolution)

                for beat in beats:
                    cur_idx = np.argmax(time_data_matrix[:,2] > beat) - 1
                    cur_time = time_data_matrix[cur_idx, 0]
                    cur_beat_len = time_data_matrix[cur_idx, 1]
                    cur_beat = time_data_matrix[cur_idx, 2]

                    if cur_beat == beat:
                        timestamps.append(cur_time)
                    else:
                        beat_dif = beat - cur_beat
                        beat_time = cur_time + cur_beat_len * beat_dif
                        timestamps.append(beat_time)
                return None

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
        tempo_change_times, tempi = self.get_tempo_changes()

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
                next_beat = beats[-1] + 60.0/tempi[n]
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
                           next_beat + beat_remaining*60.0/tempi[n] >=
                           tempo_change_times[n + 1]):
                        # Compute the amount the beat location overshoots
                        overshot_ratio = (tempo_change_times[n + 1] -
                                          next_beat)/(60.0/tempi[n])
                        # Add in the amount of the beat during this tempo
                        next_beat += overshot_ratio*60.0/tempi[n]
                        # Less of the beat remains now
                        beat_remaining -= overshot_ratio
                        # Increment the tempo index
                        n = n + 1
                    next_beat += beat_remaining*60./tempi[n]
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
        piano_roll = np.zeros((128, np.max([p.shape[1] for p in piano_rolls])),
                              dtype=np.int16)
        # Sum each piano roll into the aggregate piano roll
        for roll in piano_rolls:
            piano_roll[:, :roll.shape[1]] += roll
        return piano_roll

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
