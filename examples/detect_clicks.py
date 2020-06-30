import os
import sys
import pyhydrophone as pyhy

from pyporcc import click_detector


############## CONFIG

# Sound Files
sound_folder_path = "C:/Users/cleap/Documents/Data/Sound Data/Seiche/AutonautTest"


# Hydrophone 
name = 'Seiche'
model = 'uPam'
sensitivity = -196.0
preamp_gain = 0.0
Vpp = 2.0
hydrophone = pyhy.Seiche(name, model, sensitivity, preamp_gain, Vpp)


# Filters parameters
lowcutfreq = 100e3              # Lowcut frequency 
highcutfreq = 160e3             # Highcut frequency
min_separation_time = 0.1


# Click model
click_model_path = 'pyporcc/data/standard_click.wav'

# Output path for the detected clicks
clicks_output_path = 'C:/Users/cleap/Documents/Data/Sound Data/Clicks/autonaut_clicks.pkl'


if __name__ == "__main__":
    """
    Detect clicks on sound data
    """
    # Run on sound data
    cd = click_detector.ClickDetector()
    cd.get_click_clips(hydrophone, sound_folder_path)

    # Convert the sound clips to click and save
    clicks_df = cd.clicks_df(click_model_path)
    converter = click_detector.ClickConverter(click_model_path)
    clicks = converter.clicks_df(clicks_df, save=True, save_path=clicks_output_path)