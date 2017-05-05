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

    #prepend,append zeros so we can acknowledge inital and ending events
    piano_roll = np.hstack((np.zeros((notes, 1)),
                            piano_roll,
                            np.zeros((notes, 1))))

    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    current_velocities = np.zeros(notes,dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / sf
        if velocity > 0:
            if current_velocities[note] == 0:
                #print('note {} on'.format(pretty_midi.note_number_to_name(note)))
                #print('starting at time {} with velocity {}'.format(time,velocity))
                note_on_time[note] = time
                current_velocities[note] = velocity
            elif current_velocities[note] > 0:
                #change velocity with a special MIDI message
                pass
        else:
            #print('note {} off'.format(pretty_midi.note_number_to_name(note)))
            #print('ending at time {}'.format(time))
            pm_note = pretty_midi.Note(
                    velocity=current_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
            instrument.notes.append(pm_note)
            current_velocities[note] = 0
    pm.instruments.append(instrument)
    return midi
