
from pathlib import Path
import argparse

import pyhydrophone as pyhy
import matplotlib.pyplot as plt
from pyporcc import click_detector
from pyporcc import porcc

def run_detect_clicks():
    parser = argparse.ArgumentParser(description="""Example function to detect clicks in sound files. 
    It outputs a .csv file with times at which clicks were detected and two explanatory plots""")
    parser.add_argument('-sfil', '--soundfile',
                        type=Path,
                        help='Path to sound file (.wav file)')
    parser.add_argument('-o', '--output',
                        type=Path,
                        help='Path to folder where results will be saved')
    parser.add_argument('-sfol', '--soundfolder',
                        type=Path,
                        help='Path to folder where sound files are located')
    parser.add_argument('--lowcutfreq',
                        default=100e3,
                        type=int,
                        help='Low cut frequency. Default is 100e3')
    parser.add_argument('--highcutfreq',
                        default=160e3,
                        type=int,
                        help='High cut frequency. Default is 160e3')
    parser.add_argument('--min_separation_time',
                        default=0.1,
                        type=float,
                        help='Minimum separation time between clicks. Default is 0.1 s')

    args = parser.parse_args()

    ## Put arguments in variables
    # Paths to files and saving folders
    sound_file = args.soundfile
    save_folder = args.output
    sound_folder = args.soundfolder
    include_subfolders = True

    # Hydrophone
    model = 'ST300HF'
    name = 'SoundTrap'
    serial_number = 738496579
    soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

    # Filters parameters
    lowcutfreq = args.lowcutfreq  # Lowcut frequency
    highcutfreq = args.highcutfreq  # Highcut frequency
    min_separation_time = args.min_separation_time

    # Filter data
    prefilter = click_detector.Filter(filter_name='butter', order=4,
                                      frequencies=[100000, 150000], filter_type='bandpass')
    dfilter = click_detector.Filter(filter_name='butter', order=2,
                                    frequencies=20000, filter_type='high')
    classifier = porcc.PorCC(load_type='manual', config_file='default')

    # Run on sound data
    cd = click_detector.ClickDetector(hydrophone=soundtrap, save_folder=save_folder, save_max=10000, convert=True,
                                      classifier=classifier, dfilter=dfilter, prefilter=prefilter, save_noise=False)
    cd.detect_click_clips_folder(sound_folder, blocksize=10 * 57600)
    df_py = cd.clips
    df2 = classifier.classify_matrix(df_py)

    # Plots
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


if __name__ == "__main__":
    """
    Detect clicks on sound data
    """
    run_detect_clicks()