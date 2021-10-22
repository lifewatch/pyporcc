"""
PyPorCC

========

PyPorCC module. Classifier for harbour porpoises clicks

Detection of NBHF clicks of harbour porpoises using SoundTrap.

These NBHF signals have comparable durations around 100
μsec, high directionality, centre frequencies around 130 kHz, and
source levels generally well below 200 dB re 1 μPa (Au 1993; Madsen
et al. 2005; Villadsgaard et al. 2007; Kyhn et al. 2009). 

Clicks were only selected when the signal-to-noise ratio (SNR), given by the ratio
between the rms power of the signal of interest and the ambient
noise level measured in the same frequency band as the signal, was
at least 10 dB. Also, the first and last clicks in a click train were
identified if possible, to include as much of the entire click train as
possible in the analysis.

Prior to analysis, the sound files were filtered with a high-pass
digital Butterworth filter (4th order, 100 kHz –3dB cut off frequency). 

Click repetition rates were either measured using an automated click
detection algorithm made in Matlab or, for long ICI click trains, by
computing the pulse repetition spectrum (Watkins 1967). 


The duration of individual clicks ranges from
50 ls to 175 ls and the half-power (3 dB) bandwidth is
around 15 kHz (Kyhn et al., 2010). Clicks are emitted in
series, often referred to as “trains.” A click train is loosely
defined as “any series of clicks separated by gradually or
cyclically changing inter-click interval suggesting a unit during an echolocation event or a communication signal”

"""
from pyporcc.click_detector import ClickDetector, ClickDetectorSoundTrapHF, Filter
from pyporcc.click_converter import Click, ClickConverter
from pyporcc.porpoise_classifier import PorpoiseClassifier
from pyporcc.porcc import PorCCModel, PorCC
