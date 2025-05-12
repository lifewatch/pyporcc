import pathlib
import unittest
import pyhydrophone as pyhy

from pyporcc import ClickDetectorSoundTrapHF, ClickDetector, PorCC, Filter


# PAMGuard comparison
pamguard_output = pathlib.Path('./tests/test_data/clicks_test_pyporcc.sqlite3')
sound_folder_soundtrap = pathlib.Path("./../tests/test_data/soundtrap")
sound_folder_continuous = pathlib.Path("./../tests/test_data/continuous")
save_folder = pathlib.Path('./../tests/test_data/output')

# Hydrophone
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
# Comment the one that corresponds to your instrument
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)
soundtraphf = pyhy.soundtrap.SoundTrapHF(name=name, model=model, serial_number=serial_number)


# Hydrophone
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 5293
# Comment the one that corresponds to your instrument
soundtraphf_freq_cal = pyhy.soundtrap.SoundTrapHF(name=name,
                                                  model=model,
                                                  serial_number=serial_number,
                                                  calibration_file=pathlib.Path('./../tests/test_data/ST5293.csv'),
                                                  sep=';',
                                                  val_col_id=2)

# Filters parameters
lowcutfreq = 100e3              # Lowcut frequency
highcutfreq = 160e3             # Highcut frequency

# Define the filters
pfilter = Filter(filter_name='butter', filter_type='bandpass', order=4,
                 frequencies=[lowcutfreq, highcutfreq])
dfilter = Filter(filter_name='butter', filter_type='high', order=4, frequencies=20000)
classifier = PorCC(load_type='manual', config_file='default')


class TestDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.cd_continous = ClickDetector(hydrophone=soundtrap, save_folder=save_folder, convert=True,
                                          classifier=classifier, dfilter=dfilter, prefilter=pfilter, save_noise=False)
        self.cd_soundtrap = ClickDetectorSoundTrapHF(hydrophone=soundtraphf, save_folder=save_folder, convert=True,
                                                     classifier=classifier, prefilter=pfilter, save_noise=False)

    def test_continous(self):
        self.cd_continous.detect_click_clips_folder(sound_folder_continuous, blocksize=60 * 576000)

    def test_continous_save_max(self):
        self.cd_continous.save_max = 100
        self.cd_continous.detect_click_clips_folder(sound_folder_continuous, blocksize=10 * 576000)

    def test_soundtrap(self):
        self.cd_soundtrap.detect_click_clips_folder(sound_folder_soundtrap, blocksize=60 * 576000)
