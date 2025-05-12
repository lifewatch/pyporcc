#!/usr/bin/python
import numba as nb
import numpy as np


REF = 1.0


@nb.jit(nopython=True)
def aic_score(y, y_prob, n_features):
    """
    Return the AIC score
    """
    llk = np.sum(y * np.log(y_prob[:, 1]) + (1 - y) * np.log(y_prob[:, 0]))
    aic = 2 * n_features - 2 * llk
    return aic


@nb.jit(nopython=True)
def to_upa(wave, sensitivity, preamp_gain, Vpp):
    """
    Return the wave in db
    Parameters
    ----------
    wave
    sensitivity
    preamp_gain
    Vpp

    Returns
    -------
    The same wave converted to upa
    """
    mv = 10 ** (sensitivity / 20.0) * REF
    ma = 10 ** (preamp_gain / 20.0) * REF
    gain_upa = (Vpp / 2.0) / (mv * ma)
    return wave * gain_upa


@nb.jit(nopython=True)
def amplitude_db(clip):
    """
    Return the amplitude of the clip
    Parameters
    ----------
    clip : np.array
        Signal
    Returns
    -------
    The maximum amplitude of the clip in db
    """
    return to_db(np.max(np.abs(clip)))


@nb.jit(nopython=True)
def to_db(wave):
    """
    Convert the wave to db
    Parameters
    ----------
    wave : np.array
        wave to compute

    Returns
    -------
    wave in db
    """
    return 10 * np.log10(wave ** 2)


def constrain(x, lower, upper):
    """
    MATLAB constrain
    Constrain between upper and lower limits, and do not ignore NaN
    """
    if isinstance(x, float):
        if x < lower:
            x = lower
        elif x > upper:
            x = upper
    else:
        x[np.where(x < lower)[0]] = lower
        x[np.where(x > upper)[0]] = upper

    return x
