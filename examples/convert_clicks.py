

from pyporcc import click_detector

import pyhydrophone as pyhy


##############
# CONFIG

# Sound Files
sound_folder_path = "//archive/other_platforms/soundtrap/2017/najaar2017_reefballs_Belwind/67416073.170809"

name = 'SoundTrap'
model = 1
serial_number = 67416073
Vpp = 2
hydrophone = pyhy.SoundTrapHF(name, model, serial_number, Vpp)


# Click model
click_model_path = '../pyporcc/data/standard_click.wav'

# Output path for the detected clicks
clicks_output_path = "test.csv"


if __name__ == "__main__":
    """
    Detect clicks on sound data
    """
    # Convert the sound clips to click and save
    clicks_df = hydrophone.read_HFclicks(sound_folder_path)
    converter = click_detector.ClickConverter(click_model_path)
    clicks = converter.clicks_df(clicks_df, save=True, save_path=clicks_output_path)
