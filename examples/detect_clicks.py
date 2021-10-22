import pathlib
import pyhydrophone as pyhy

from pyporcc import ClickDetectorSoundTrapHF, ClickDetector, PorCC, Filter


sound_folder = pathlib.Path("./../tests/test_data/soundtrap")
save_folder = pathlib.Path('./../tests/test_data/output')

# Hydrophone
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrapHF(name=name, model=model, serial_number=serial_number)

# Filters parameters
lowcutfreq = 100e3              # Lowcut frequency
highcutfreq = 160e3             # Highcut frequency

# Define the filters
pfilter = Filter(filter_name='butter', filter_type='bandpass', order=4,
                                frequencies=[lowcutfreq, highcutfreq])
dfilter = Filter(filter_name='butter', filter_type='high', order=4, frequencies=20000)
classifier = PorCC(load_type='manual', config_file='default')

cd = ClickDetectorSoundTrapHF(hydrophone=soundtrap, save_folder=save_folder, convert=True,
                              classifier=classifier, prefilter=pfilter, save_noise=False)


if __name__ == "__main__":
    """
    Detect clicks on sound data
    """
    cd.detect_click_clips_folder(sound_folder, blocksize=60 * 576000)
