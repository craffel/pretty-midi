"""
Play all the 3-note chords on one octave, after Tom Johnson's "Chord Catalog".
Example originally by Douglas Repetto
"""

import pretty_midi
import itertools

# Our starting note/octave is middle C
base_note = 60

# Time between each chord
chord_duration = .1
# Length of each note
note_duration = chord_duration*.8

# Make a pretty midi object
pm = pretty_midi.PrettyMIDI()

# Add synth voice instrument
synth_voice = pretty_midi.instrument_name_to_program('Whistle')
pm.instruments.append(pretty_midi.Instrument(synth_voice))

# Keep track of timing
curr_time = 0.0
# All notes have velocity 100
velocity = 100

# itertools.combinations computes all pairs of items without replacement
for offset_1, offset_2 in itertools.combinations(range(1, 12), 2):
    # Create our chord from our three chord[n] values
    # Notes start at curr_time and end at curr_time + note_duration
    pm.instruments[0].notes.append(pretty_midi.Note(
        velocity, base_note, curr_time, curr_time + note_duration))
    pm.instruments[0].notes.append(pretty_midi.Note(
        velocity, base_note + offset_1, curr_time, curr_time + note_duration))
    pm.instruments[0].notes.append(pretty_midi.Note(
        velocity, base_note + offset_2, curr_time, curr_time + note_duration))
    # Increment curr_time with chord_duration
    curr_time += chord_duration

midi_filename = "all_chords.mid"
pm.write(midi_filename)
