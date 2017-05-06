from __future__ import division
"""
Convert Piano Roll back into PrettyMIDI object.
Most of this code is from @carlthome.
@jsleep Adapted that code from mido to pretty_midi.
"""
import pretty_midi
import numpy as np



def piano_roll_to_pretty_midi(piano_roll, sf=100, program_num=1):
    """Convert piano roll to a single instrument pretty_midi object"""
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program_num)

    # prepend,append zeros so we can acknowledge inital and ending events
    piano_roll = np.hstack((np.zeros((notes, 1)),
                            piano_roll,
                            np.zeros((notes, 1))))

    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    current_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / sf
        if velocity > 0:
            if current_velocities[note] == 0:
                 #print('note {} on'.format(
                 #  pretty_midi.note_number_to_name(note)))
                 #print('starting at time {} with velocity {}'.format(
                 #time,velocity))
                note_on_time[note] = time
                current_velocities[note] = velocity
            elif current_velocities[note] > 0:
                # change velocity with a special MIDI message
                pass
        else:
            # print('note {} off'.format(
            #  pretty_midi.note_number_to_name(note)))
            # print('ending at time {}'.format(time))
            pm_note = pretty_midi.Note(
                    velocity=current_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
            instrument.notes.append(pm_note)
            current_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

# test example

# Create a PrettyMIDI object
cello_c_am = pretty_midi.PrettyMIDI()
# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)
# Iterate over C Major
for note_name in ['C5', 'E5', 'G5']:
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance, starting at 0s and ending at .5s
    note = pretty_midi.Note(
        velocity=100, pitch=note_number, start=0, end=.5)
    # Add it to our cello instrument
    cello.notes.append(note)
# Iterate over A minor
for note_name in ['A4', 'E5', 'C5']:
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance, starting at 0s and ending at .5s
    note = pretty_midi.Note(
        velocity=100, pitch=note_number, start=.5, end=1)
    # Add it to our cello instrument
    cello.notes.append(note)
# Add the cello instrument to the PrettyMIDI object
cello_c_am.instruments.append(cello)
# Get piano Roll
cello_pr = cello_c_am.get_piano_roll()
# Piano Roll back to PrettyMidi
new_cello = piano_roll_to_pretty_midi(cello_pr,
             program_num=pretty_midi.instrument_name_to_program('Cello'))
new_cello.write('cello-C-Am.mid')
