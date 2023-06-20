"""Utility functions for handling fluidsynth

"""

import os
import pkg_resources

try:
    import fluidsynth
    _HAS_FLUIDSYNTH = True
except ImportError:
    _HAS_FLUIDSYNTH = False

DEFAULT_SF2 = 'TimGM6mb.sf2'


def get_fluidsynth_instance(synthesizer=None, sfid=0, fs=44100):
    """ Check if a valid fluidsynth.Synth instance is provided, and if not,
    create one.

    Parameters
    ----------
    synthesizer : fluidsynth.Synth or str
        fluidsynth.Synth instance to use or a string with the path to a .sf2 file.
        Default ``None``, which creates a new instance using the TimGM6mb.sf2 file
        included with ``pretty_midi``.
    sfid : int
        Soundfont ID to use if an instance of fluidsynth.Synth is provided.
        Default ``0``, which uses the first soundfont.
    fs : int
        Sampling rate to synthesize at.
        Only used when a new instance of fluidsynth.Synth is created.
        Default ``44100``.

    Returns
    -------
    synthesizer : fluidsynth.Synth
        fluidsynth.Synth instance
    sfid : int
        Soundfont ID
    new_instance_created : bool
        Whether a new instance of fluidsynth.Synth was created.

    """
    if not _HAS_FLUIDSYNTH:
        raise ImportError("fluidsynth() was called but pyfluidsynth is not installed.")

    if synthesizer is None:
        synthesizer = pkg_resources.resource_filename(__name__, DEFAULT_SF2)

    # Create a fluidsynth instance if one wasn't provided
    if isinstance(synthesizer, str):
        sf2_path = synthesizer
        if not os.path.exists(synthesizer):
            raise ValueError("No soundfont file found at the supplied path {}".format(sf2_path))
        synthesizer = fluidsynth.Synth(samplerate=fs)
        sfid = synthesizer.sfload(sf2_path)
        new_instance_created = True
    elif isinstance(synthesizer, fluidsynth.Synth):
        new_instance_created = False
    else:
        raise ValueError("synthesizer must be a str or a fluidsynth.Synth instance")

    return synthesizer, sfid, new_instance_created
