#!/usr/bin/python
"""
Module : click_converter.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Instituut voor de Zee)
Last Accessed : 9/23/2020
"""

__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

import sys
from importlib import resources

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import soundfile as sf
from scipy import interpolate
from scipy import signal as sig


class ClickConverter:
    def __init__(self, fs, click_model_path=None, click_vars=None):
        """
        Init the click converter

        Parameters
        ----------
        click_model_path : string or Path
            File where the click model is stored
        click_vars : list of strings
            List of the output parameters to compute for each click
        """
        if click_model_path is None:
            with resources.path('pyporcc.data', 'standard_click.wav') as click_model_path:
                print('Setting the click model path to default...')
                self.click_model_path = click_model_path
        self.click_model, self.fs_model = sf.read(click_model_path)
        self.fs = fs

        if click_vars is None:
            self.click_vars = ['Q', 'duration', 'ratio', 'XC', 'CF', 'BW']
        else:
            self.click_vars = click_vars

    def __setattr__(self, key, value):
        if key == 'fs':
            if self.fs_model != value:
                click_model, fs_model = sf.read(self.click_model_path)
                print('This click is not recorded at the same frequency than the classified data! '
                      'Resampling to %s S/s' % value)
                new_samples = int(np.ceil(click_model.size / fs_model * value))
                self.click_model = sig.resample(click_model, new_samples)
                self.fs_model = value
        self.__dict__[key] = value

    def clicks_df(self, df, nfft=512, save_path=None):
        """
        Find all the clicks and return them as a df with all the clicks params

        Parameters
        ----------
        df : DataFrame
            It needs to have at least datetime and wave
        nfft : int
            Length of FFT
        save_path : string or Path
            Path to the desired save file. If None, it is not saved
        """
        # Calculate all the independent variables and append them to the df
        params_matrix = np.zeros((len(df), len(self.click_vars)))

        for idx in df.index:
            row = df.iloc[idx]
            params_matrix[idx, :] = self.click_params(row['wave'], nfft=nfft)

        df[self.click_vars] = params_matrix

        # Keep the metadata
        df.fs = df.fs

        if save_path is not None:
            extension = save_path.split('.')[-1]
            if extension == 'json':
                df.to_json(save_path)
            elif extension == 'pkl':
                df.to_pickle(save_path)
            elif extension == 'csv':
                df.drop(['wave'], axis=1).to_csv(save_path)
            elif extension == 'hdf5':
                df.to_hdf(save_path, mode='w', key='clips')
            else:
                raise Exception('The extension %s is unkown' % extension)

        return df

    def row2click(self, row):
        """
        Convert the row to an object Click
        Parameters
        ----------
        row : Pandas DataFrame row

        Returns
        -------
        Click object
        """
        signal = row['wave']
        if 'datetime' in row.axes[0]:
            timestamp = row.datetime
        else:
            timestamp = row.name
        click = Click(signal, self.fs, timestamp, click_model_path=self.click_model_path, verbose=False)
        return click

    def add_params_to_row(self, row, nfft=512):
        row.loc[self.click_vars] = self.click_params(row['wave'], nfft)
        return row

    def click_params(self, sound_block, nfft=512):
        q, duration, ratio, xc, cf, bw, _, _ = click_params(sound_block, self.fs, self.click_model, nfft)
        return q, duration, ratio, xc, cf, bw

    @staticmethod
    def test_click_calculation(df_clicks, df_test, col_vars):
        """
        Test the calculation of the click parameters indicated in col_vars obtained with python
        compared to the ones obtained in the paper (on the Test DB)
        """
        # Compare each field
        rel_error = np.abs(df_test[col_vars] - df_clicks[col_vars]) / df_test[col_vars].mean()
        mean_rel_error = rel_error.mean()

        return mean_rel_error


@nb.njit
def zero_pad(sound_block, nfft):
    """
    Return a zero-padded sound block
    Parameters
    ----------
    sound_block: np.array
        Sound clip representing the possible click
    nfft : int
        Desired length
    """
    if len(sound_block) < nfft:
        zero_padded = np.zeros(nfft)
        zero_padded[0:sound_block.size] = sound_block
        sound_block = zero_padded
    return sound_block


@nb.jit(nopython=True)
def fast_click_params(sound_block, fs, psd, freq):
    """
    Return the duration of the 80% of the energy of the sound block
    Parameters
    ----------
    sound_block: np.array
        Sound Block to analyze
    fs : int
        Sampling Frequency
    psd : np.array
        Power Spectral Density
    freq : np.array
        Frequencies of the psd

    Returns
    -------
    Duration in microseconds
    """
    ener = np.cumsum(sound_block ** 2)
    istart_list = np.where(ener <= (ener[-1] * 0.1))[0]  # index of where the 10% is
    iend_list = np.where(ener <= (ener[-1] * 0.9))[0]  # index of where the 90% is
    if len(istart_list) > 0:
        istart = istart_list[-1]
    else:
        istart = 0
    if len(iend_list) > 0:
        iend = iend_list[-1]
    else:
        iend = len(ener)
    # duration in microseconds
    duration = iend - istart
    duration = duration / fs * 1e6

    # Normalize spectrum
    psd = psd / np.max(psd)
    cf = np.sum(freq * (psd ** 2)) / np.sum(psd ** 2)
    pf = freq[psd.argmax()]

    # Calculate RMSBW
    rmsbw = (np.sqrt((np.sum(((freq - cf) ** 2) * (psd ** 2))) / np.sum(psd ** 2))) / 1000.0

    # Parameters according to Mhl & Andersen, 1973
    q = (cf / rmsbw) / 1000.0
    ratio = pf / cf

    return duration, cf, pf, q, ratio


def powerbw(x, fs):
    """
    Calculate -3dB bandwidth imitating MATLAB powerbw
    Parameters
    ----------
    x : np.array
        Signal to process
    fs : int
        Sampling frequency

    Returns
    -------
    -3dB Bandwidth in kHz
    """
    beta = 0
    freq, psd = sig.periodogram(x=x, fs=fs, window=('kaiser', beta), scaling='spectrum',
                                return_onesided=False)

    # Consecutive frequencies of the psd that have more than half of the maximum freq power
    r = 10 * np.log10(0.5)
    half = np.max(psd) * (10 ** (r / 10))
    i_center = psd.argmax()

    i_l = np.where(psd[0:i_center] <= half)[0]
    i_r = np.where(psd[i_center:-1] <= half)[0]

    if len(i_l) == 0:
        f_lo = freq[0]
    else:
        i_l = i_l[-1]
        psd_points = np.log10([max(psd[i_l], sys.float_info.min), max(psd[i_l+1], sys.float_info.min)])
        inter = interpolate.interp1d(psd_points, freq[i_l: i_l + 2])
        f_lo = inter(np.log10(half))

    if len(i_r) == 0:
        f_hi = freq[-1]
    else:
        i_r = i_r[0] + i_center
        psd_points = np.log10([max(psd[i_r - 1], sys.float_info.min), max(psd[i_r], sys.float_info.min)])
        inter = interpolate.interp1d(psd_points, freq[i_r - 1: i_r + 1])
        f_hi = inter(np.log10(half))

    bw = (f_hi - f_lo) / 1000.0

    return bw


def click_params(sound_block, fs, click_model, nfft):
    """
    Return the click parameters necessary to classify the click using PorCC
    Parameters
    ----------
    sound_block
    fs
    click_model
    nfft

    Returns
    -------
    Q, duration, ratio, XC, CF, BW, psd and freq
    """
    # Calculate PSD, freq, centrum-freq (cf), peak-freq (pf) of the sound file
    sound_block_padded = zero_pad(sound_block, nfft)
    freq, psd = sig.periodogram(x=sound_block_padded, fs=fs, nfft=nfft, scaling='spectrum')
    duration, cf, pf, q, ratio = fast_click_params(sound_block, fs, psd, freq)
    bw = powerbw(sound_block, fs)

    # Calculate the correlation with the model
    x_coeff = np.correlate(sound_block, click_model, mode='same')
    xc = np.max(x_coeff)

    return q, duration, ratio, xc, cf, bw, psd, freq


class Click:
    def __init__(self, sound_block, fs, timestamp, click_model_path, nfft=512, verbose=False):
        """
        Start a porpoise click object

        Parameters
        ----------
        sound_block : np.array
            Clip of the click
        fs : int
            Sampling Frequency
        timestamp : datetime
            Timestamp where the click starts
        click_model_path : string or Path
            File where the click model is stored
        nfft : int
            Number of frequency bins to use for the spectrogram calculation
        verbose : bool
            Set to True if plots are desired
        """
        self.nfft = nfft
        self.fs = fs
        self.sound_block = sound_block
        self.timestamp = timestamp
        self.click_model, fs_model = sf.read(click_model_path)
        if fs_model != fs:
            if verbose:
                print('This click is not recorded at the same frequency than the classified data! '
                      'Resampling to %s S/s' % self.fs)
            new_samples = int(np.ceil(self.click_model.size / fs_model * self.fs))
            self.click_model = sig.resample(self.click_model, new_samples)
        else:
            self.fs = fs

        self.q, self.duration, self.ratio, self.xc, self.cf, self.bw, psd, self.freq = click_params(sound_block,
                                                                                                    self.fs,
                                                                                                    self.click_model,
                                                                                                    nfft)
        if verbose:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(self.sound_block)
            ax[1].plot(self.freq, psd)
            plt.show()
            plt.close()

    def __getattribute__(self, name):
        """
        Avoid confusion with capital and low letters
        """
        return super().__getattribute__(name.lower())
