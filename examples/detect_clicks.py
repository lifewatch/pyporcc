
import pathlib
import pandas as pd
import pyhydrophone as pyhy
import matplotlib.pyplot as plt

from pyporcc import click_detector
from pyporcc import porcc

# CONFIG

# Sound Files
sound_folder = pathlib.Path('//fs/SHARED/transfert/#JOBSTUDENTS#/2020/Javier/pyporcc/test')
sound_file = pathlib.Path('//fs/SHARED/transfert/#JOBSTUDENTS#/2020/Javier/pyporcc/test/738496579.150824180131.wav')
save_folder = pathlib.Path('//fs/SHARED/transfert/#JOBSTUDENTS#/2020/Javier/pyporcc/test/save')
include_subfolders = True

# Hydrophone
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 738496579
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# Filters parameters
lowcutfreq = 100e3              # Lowcut frequency
highcutfreq = 160e3             # Highcut frequency
min_separation_time = 0.1


if __name__ == "__main__":
    """
    Detect clicks on sound data
    """
    prefilter = click_detector.Filter(filter_name='butter', order=4,
                                      frequencies=[100000, 150000], filter_type='bandpass')
    dfilter = click_detector.Filter(filter_name='butter', order=2,
                                    frequencies=20000, filter_type='high')
    classifier = porcc.PorCC(load_type='manual', config_file='default')
    # Run on sound data
    cd = click_detector.ClickDetector(hydrophone=soundtrap, save_folder=save_folder, save_max=10000, convert=True,
                                      classifier=classifier, dfilter=dfilter, prefilter=prefilter, save_noise=False)
    cd.detect_click_clips_folder(sound_folder, blocksize=10*57600)
    df_py = cd.clips
    df2 = classifier.classify_matrix(df_py)

    fig, ax = plt.subplots(1, 1, sharex='all')
    ax.scatter(df_py.start_sample, df_py.amplitude, s=2.0)
    ax.set_title('pyporcc detections')
    ax.set_ylabel('Amplitude rms [db]')
    ax.set_xlabel('Samples')
    plt.show()

    fig, ax = plt.subplots(1, 1, sharex='all')
    ax.scatter(df_py.start_sample, df_py.duration_samples, s=2.0)
    ax.set_title('pyporcc detections')
    ax.set_ylabel('Duration [samples]')
    ax.set_xlabel('Samples')
    plt.show()