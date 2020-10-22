
import pathlib
import sqlite3
import pandas as pd
import pyhydrophone as pyhy
import matplotlib.pyplot as plt

from pyporcc import click_detector


# CONFIG

# Sound Files
sound_file_path = pathlib.Path("../pyporcc/data/738496579.150824180131.wav")
save_folder = 'C:/Users/cleap/Documents/Data/Clicks/pyporcc/test/'
include_subfolders = True

# PAMGuard comparison
pamguard_output = 'C:/Users/cleap/Documents/Data/Clicks/PAMGuard/clicks_test_pyporcc_4.sqlite3'

# Hydrophone 
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 738496579
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# Filters parameters
lowcutfreq = 100e3              # Lowcut frequency 
highcutfreq = 160e3             # Highcut frequency
min_separation_time = 0.1

# Click model
click_model_path = '../pyporcc/data/standard_click.wav'


if __name__ == "__main__":
    """
    Detect clicks on sound data
    """
    # Run on sound data
    cd = click_detector.ClickDetector(hydrophone=soundtrap, save_folder=save_folder)
    cd.detect_click_clips_file(sound_file_path, blocksize=60*500000)
    df_py = cd.clips[['start_sample', 'amplitude']]

    # Read the PAMGuard output to compare
    conn = sqlite3.connect(pamguard_output)
    query = "select * from Click_Detector_Clicks"
    df_pamguard = pd.read_sql_query(query, conn)
    df_pamguard = df_pamguard[['UTC', 'startSample', 'amplitude']]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].scatter(df_pamguard.startSample, df_pamguard.amplitude, s=2.0)
    ax[1].scatter(df_py.start_sample, df_py.amplitude, s=2.0)
    ax[0].set_title('PAMGuard detections')
    ax[1].set_title('pyporcc detections')
    ax[0].set_ylabel('Amplitude rms [db]')
    ax[1].set_ylabel('Amplitude rms [db]')
    ax[1].set_xlabel('Samples')
    plt.show()
    print('hello!')
    # Convert the sound clips to click and save
    # converter = click_detector.ClickConverter(click_model_path)
    # clicks = converter.clicks_df(df_py, save=True, save_path=clicks_output_path)
