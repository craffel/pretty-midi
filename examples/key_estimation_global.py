# util
import os, csv
import glob2 as glob
import numpy as np

import pretty_midi as pm
import estimators as est

filepath = "../example.mid"

# predict global key using published algorithms by Krumhansl and Schmucker,
# Madsen and Widmer, and Temperley.

pm_data = pm.PrettyMIDI(str(filepath))

# model parameters
algorithm = 'krumhansl_schmucker'
distance = 'correlation'
key_profile_id = 'TKP'

# remove pitch bends
for i in xrange(len(pm_data.instruments)):
    pm_data.instruments[i].pitch_bends = []

# get pitch class histogram, pitch class transition matrix
# and beat-aligned chromagram
histogram = pm_data.get_pitch_class_histogram(use_duration=True,
                                              use_velocity=True,
                                              normalize=True)
transition_matrix = pm_data.get_pitch_class_transition_matrix(normalize=True)
beats = pm_data.get_beats()
chromagram = pm_data.get_chroma(times=beats)

key_pc = est.estimate_key(histogram,
                          algorithm,
                          distance,
                          key_profile_id)

algorithm = 'madsen_widmer'
distance = 'sum_of_products' # not necessary
key_profile_id = 'SG'
key_interval = est.estimate_key(transition_matrix,
                                algorithm,
                                distance,
                                key_profile_id)

algorithm = 'temperley'
distance = 'temperley'
key_profile_id = 'TKP'
key_temperley = est.estimate_key(chromagram,
                                algorithm,
                                distance,
                                key_profile_id)
key_temperley = (key_temperley[0], key_temperley[1], key_temperley[2])

print """{}
    pitch_class \t{}
    interval \t{}
    temperley \t{}""".format(os.path.basename(filepath), key_pc,
        key_interval, key_temperley)
