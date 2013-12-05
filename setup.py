from setuptools import setup

setup(
    name='pretty_midi',
    version='0.0.1',
    description='A class for handling MIDI data in a convenient way.',
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/pretty_midi',
    packages=['pretty_midi'],
    long_description="""\
    A class which makes handling MIDI data easy in Python.  Provides methods for extracting and modifying the useful parts of MIDI files.
    """,
    classifiers=[
          "License :: OSI Approved :: GNU General Public License (GPL)",
          "Programming Language :: Python",
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    keywords='audio music midi mir',
    license='GPL',
    install_requires=[
        'numpy >= 1.7.0',
    ],
)
