from setuptools import setup

setup(
    name='pretty_midi',
    version='0.2.2',
    description='Functions and classes for handling MIDI data conveniently.',
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/pretty_midi',
    packages=['pretty_midi'],
    package_data={'': ['TimGM6mb.sf2']},
    long_description="""\
    Functions and classes which make handling MIDI data easy in Python.
    Provides methods for parsing, modifying, and analyzing MIDI files.
    """,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
    ],
    keywords='audio music midi mir',
    license='MIT',
    install_requires=[
        'numpy >= 1.7.0',
        'midi'
    ],
)
