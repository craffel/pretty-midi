pretty-midi contains utility function/classes for handling MIDI data, so that it's in a format which is easy to modify and extract information from.

As of now it relies on the python-midi package:

https://github.com/vishnubob/python-midi/

Example usage:

```python
from pretty_midi import PrettyMIDI
bohemian_rhapsody = PrettyMIDI(midi.read_midifile('Bohemian Rhapsody.mid'))
piano_roll = bohemian_rhapsody.get_piano_roll()
import matplotlib.pyplot as plt
plt.figure( figsize=(20, 8) )
plt.imshow( piano_roll, origin='lower', aspect='auto', interpolation='nearest' )
```
