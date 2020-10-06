#!/usr/bin/python
"""
Module : utils.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Instituut voor de Zee)
Last Accessed : 9/23/2020
"""

import numpy as np
import numba as nb

@nb.jit
def aic_score(y, y_prob, n_features):
    """
    Return the AIC score
    """
    llk = np.sum(y*np.log(y_prob[:, 1]) + (1 - y)*np.log(y_prob[:, 0]))
    aic = 2*n_features - 2*llk
    return aic


def constrain(x, lower, upper):
    """
    MATLAB constrain
    Constrain between upper and lower limits, and do not ignore NaN
    """
    x[np.where(x < lower)[0]] = lower
    x[np.where(x > upper)[0]] = upper

    return x
