pretty_midi.py contains utility function/classes for handling MIDI data, so that it's in a format which is easy to modify and extract information from.

As of now it relies on the python-midi package:

https://github.com/vishnubob/python-midi/

In order to synthesize some MIDI data using the included soundfont file (see the fluidsynth() function), you need fluidsynth and pyfluidsynth:

http://www.fluidsynth.org/

https://code.google.com/p/pyfluidsynth/

Example usage:

```python
from pretty_midi import PrettyMIDI
import midi
# Construct PrettyMIDI object
bohemian_rhapsody = PrettyMIDI(midi.read_midifile('Bohemian Rhapsody.mid'))

# Get a piano roll matrix of this MIDI file, sampled at 100 Hz
piano_roll = bohemian_rhapsody.get_piano_roll()
import matplotlib.pyplot as plt
plt.figure( figsize=(20, 8) )
plt.imshow( piano_roll, origin='lower', aspect='auto', interpolation='nearest' )

# Get the chroma matrix - the energy in each semitone across octaves
chroma = bohemian_rhapsody.get_chroma()
# Optional - normalize chroma_matrix columnwise by max
chroma /= (chroma.max( axis = 0 ) + (chroma.max( axis = 0 ) == 0))
plt.figure( figsize=(20, 4) )
plt.imshow( chroma, origin='lower', aspect='auto', interpolation='nearest' )

# Iterate over the instrument tracks in the MIDI file
for instrument in bohemian_rhapsody.instruments:
    # Check whether the instrument is a drum track
    if not instrument.is_drum:
        # Iterate over note events for this instrument
        for note in instrument.events:
            # Shift them up by 4 semitones
            note.pitch += 4
# Write out key-shifted version
bohemian_rhapsody.write('Bohemian Rhapsody-Shifted.mid')
```
