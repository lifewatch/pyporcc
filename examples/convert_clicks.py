
import argparse
from pyporcc import click_detector
import pyhydrophone as pyhy

def run_detect_clicks():
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('-sfol', '--sound_folder_path',
                        type=str,
                        help='Path to sound files folder')
    parser.add_argument('-input', '--click_model_path',
                        type=str,
                        help='Path to click model')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='Output path for the detected clicks')
    args = parser.parse_args()

    sound_folder_path = args.sound_folder_path
    click_model_path = args.click_model_path
    output = args.output

## CONFIG
name = 'SoundTrap'
model = 1
serial_number = 67416073
Vpp = 2
hydrophone = pyhy.SoundTrapHF(name, model, serial_number, Vpp)

if __name__ == "__main__":
    """
    Detect clicks on sound data
    """
    # Convert the sound clips to click and save
    clicks_df = hydrophone.read_HFclicks(sound_folder_path)
    converter = click_detector.ClickConverter(click_model_path)
    clicks = converter.clicks_df(clicks_df, save_path=output)
