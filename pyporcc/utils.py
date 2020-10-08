#!/usr/bin/python
"""
Module : utils.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Instituut voor de Zee)
Last Accessed : 9/23/2020
"""

__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

import numpy as np
import numba as nb

REF = 1.0

@nb.jit
def aic_score(y, y_prob, n_features):
    """
    Return the AIC score
    """
    llk = np.sum(y*np.log(y_prob[:, 1]) + (1 - y)*np.log(y_prob[:, 0]))
    aic = 2*n_features - 2*llk
    return aic


@nb.jit
def amplitude_db(wave, sensitivity, preamp_gain, Vpp):
    """
    Calculate the amplitude
    """
    a = np.abs(wave).max()
    mv = 10 ** (sensitivity / 20.0) * REF
    ma = 10 ** (preamp_gain / 20.0) * REF
    gain_upa = (Vpp / 2.0) / (mv * ma)
    return 10*np.log10((a * gain_upa)**2 / REF)


def constrain(x, lower, upper):
    """
    MATLAB constrain
    Constrain between upper and lower limits, and do not ignore NaN
    """
    x[np.where(x < lower)[0]] = lower
    x[np.where(x > upper)[0]] = upper

    return x
