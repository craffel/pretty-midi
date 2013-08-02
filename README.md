pretty_midi.py contains utility function/classes for handling MIDI data, so that it's in a format which is easy to modify and extract information from.

As of now it relies on the python-midi package:

https://github.com/vishnubob/python-midi/

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
```
