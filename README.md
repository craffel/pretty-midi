`pretty_midi` contains utility function/classes for handling MIDI data, so that it's in a format which is easy to modify and extract information from.

Documentation is available [here](http://craffel.github.io/pretty-midi/).  You can also find a Jupyter notebook tutorial [here](http://nbviewer.jupyter.org/github/craffel/pretty-midi/blob/master/Tutorial.ipynb).

`pretty_midi` is available via [pip](https://pypi.python.org/pypi/pretty_midi) or via the [setup.py](https://github.com/craffel/pretty-midi/blob/master/setup.py) script. In order to synthesize MIDI data using fluidsynth, you need the [fluidsynth](http://www.fluidsynth.org/) program and [pyfluidsynth](https://pypi.python.org/pypi/pyfluidsynth).

If you end up using `pretty_midi` in a published research project, please cite the following report:

Colin Raffel and Daniel P. W. Ellis. [_Intuitive Analysis, Creation and Manipulation of MIDI Data with pretty_midi_](http://colinraffel.com/publications/ismir2014intuitive.pdf). In Proceedings of the 15th International Conference on Music Information Retrieval Late Breaking and Demo Papers, 2014.


Example usage for analyzing, manipulating and synthesizing a MIDI file:

```python
import pretty_midi
# Load MIDI file into PrettyMIDI object
midi_data = pretty_midi.PrettyMIDI('example.mid')
# Print an empirical estimate of its global tempo
print(midi_data.estimate_tempo())
# Compute the relative amount of each semitone across the entire song, a proxy for key
total_velocity = sum(sum(midi_data.get_chroma()))
print([sum(semitone)/total_velocity for semitone in midi_data.get_chroma()])
# Shift all notes up by 5 semitones
for instrument in midi_data.instruments:
    # Don't want to shift drum notes
    if not instrument.is_drum:
        for note in instrument.notes:
            note.pitch += 5
# Synthesize the resulting MIDI data using sine waves
audio_data = midi_data.synthesize()
```

Example usage for creating a simple MIDI file:

```python
import pretty_midi
# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()
# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)
# Iterate over note names, which will be converted to note number later
for note_name in ['C5', 'E5', 'G5']:
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance for this note, starting at 0s and ending at .5s
    note = pretty_midi.Note(velocity=100, pitch=note_number, start=0, end=.5)
    # Add it to our cello instrument
    cello.notes.append(note)
# Add the cello instrument to the PrettyMIDI object
cello_c_chord.instruments.append(cello)
# Write out the MIDI data
cello_c_chord.write('cello-C-chord.mid')
```
