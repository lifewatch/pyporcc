
import pyhydrophone as pyhy

from pyporcc import click_detector


############## CONFIG

# Sound Files
sound_folder_path = "C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/COVID-19/67416073.200427 Zeekat"


# Hydrophone 
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)


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
    clicks_df = cd.get_click_clips(soundtrap, sound_folder_path)

    # Convert the sound clips to click and save
    converter = click_detector.ClickConverter(click_model_path)
    clicks = converter.clicks_df(clicks_df, save=True, save_path=clicks_output_path)