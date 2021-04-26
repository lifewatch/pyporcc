

from pyporcc import click_detector, click_converter

import pyhydrophone as pyhy


##############
# CONFIG

# Sound Files
sound_folder_path = "//archive/other_platforms/soundtrap/2017/Belwind/najaar2017_reefballs_Belwind"

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
    clicks_df = hydrophone.read_HFfolder(sound_folder_path, zip_mode=True)
    converter = click_converter.ClickConverter(click_model_path)
    clicks = converter.clicks_df(clicks_df, save_path=clicks_output_path)
