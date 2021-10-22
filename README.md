# PyPorCC

PyPorCC is a package that allows the detection and classification of Harbor Porpoises' clicks.
The detection of clicks in continuous files is a python adaptation of the PAMGuard click detector algorithm. 
> Gillespie D, Gordon J, McHugh R, McLaren D, Mellinger DK, Redmond P, Thode A, Trinder P, Deng XY (2008) PAMGUARD: 
>Semiautomated, open source software for real-time acoustic detection and localisation of cetaceans.
> Proceedings of the Institute of Acoustics 30:54–62.

The classification is done using the PorCC algorithm, adapted to python from the paper: 
> Cosentino, M., Guarato, F., Tougaard, J., Nairn, D., Jackson, J. C., & Windmill, J. F. C. (2019). 
> Porpoise click classifier (PorCC): A high-accuracy classifier to study harbour porpoises ( Phocoena phocoena ) in the wild . 
> The Journal of the Acoustical Society of America, 145(6), 3427–3434. https://doi.org/10.1121/1.5110908

Also other models can be trained. The implemented ones so far are: 
* `svc`: Support Vector Machines
* `lsvc`: Linear Support Vector Machines
* `RandomForest`: Random Forest
* `knn`: K-Nearest Neighbor

## Usage
### Click detector
The Click detector can be used in continuous wav files (with higher than 300 kHz sampling rate) or in the SoundTrap 
HF output files (*.bcl + *.dwv). 
ForSoundTrapHF files, you can create a ClickDetectorSoundTrapHF object with the necessary parameters and run it as: 

```python 
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
cd.detect_click_clips_folder(sound_folder, blocksize=60 * 576000)

```
For continuous data, just make sure you use the class ClickDetector object instead of a ClickDetectorSoundTrapHF!
The rest of the code should be the same (except the hydrophone definition, which will depend on the instrument you use)

## Note
Please note, the clicks PAMGuard's Click Classifier classified as porpoise clicks appear as 0 in both ClassifiedAs 
and ManualAssign fields. 

## Citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5179943.svg)](https://doi.org/10.5281/zenodo.5179943)

