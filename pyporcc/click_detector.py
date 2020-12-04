#!/usr/bin/python
"""
Module : click_detector.py
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
import datetime as dt
import pathlib
import zipfile
from importlib import resources

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import interpolate
from scipy import signal as sig
from tqdm import tqdm
import h5py

from pyporcc import utils


class ClickDetector:
    def __init__(self, hydrophone=None, long_filt=0.00001, long_filt2=0.000001, short_filt=0.1, threshold=10,
                 min_separation=100, max_length=1024, min_length=90, pre_samples=40, post_samples=40, fs=576000,
                 prefilter=None, dfilter=None, save_max=np.inf, save_folder='.',
                 convert=False, click_model_path=None, classifier=None, save_noise=False):
        """
        Process to detect clicks
        Trigger decision
        The trigger automatically makes a measure of background noise and then compares the signal level to the
        noise level. When the signal level reaches a certain threshold above the noise level a click clip is started.
        When the signal level falls to below the threshold for more than a set number of bins, the click clip ends and
        the clip is sent to the localisation and classification modules.
        Parameters
        ----------
        hydrophone : hydrophone object
            Object representing a hydrophone from pyhydrophone
        long_filt : float
            Long filter (used when no click is active)
        long_filt2 : float
            Long filter 2 (used when signal is above threshold)
        short_filt: float
            Short filter parameter
        threshold : float
            Detection threshold in db
        max_length : int
            Maximum length of a click in samples
        threshold : float
            Detection threshold in db
        min_separation : int
            Minimum separation between clicks in samples
        max_length : int
            Maximum length of a click in samples
        pre_samples : int
            Number of samples to store before the click detection
        post_samples : int
            Number of samples to store after the click detection
        fs : int
            Sampling frequency
        prefilter: Filter object
            Prefilter to apply to the signal before passing it to the trigger. It is also applied to the stored clips.
            If set to None a band-pass Butterworth 4th order filter [100000, 150000] Hz will be used
        dfilter : Filter object
            Filter to apply for the trigger. Has to be of the class Filter
            If set to None, a high-pass Butterworth [20000, :] 4th order filter will be used
        save_max : int
            Maximum number of clicks to save in a file
        save_folder : string or Path
            Folder where to save the click output
        classifier : classifier object
            Classifier used to classify the snippets. Needs to have a classify_click function
        save_noise : bool
            Set to True if you want to save the noise clips as well
        """
        self.hydrophone = hydrophone
        # Detector parameters
        self.pre_samples = pre_samples
        self.post_samples = post_samples

        real_min_length = min_length - post_samples - pre_samples
        real_max_length = max_length - post_samples - pre_samples
        if real_min_length < 0:
            real_min_length = 1
            print('This min length is less than pre_samples + post_samples. Setting it to 1...')
        if real_max_length < 0:
            real_max_length = 1000
            print('This max length is less than pre_samples + post_samples. Setting it to 1000...')
        self.triggerfilter = TriggerFilter(long_filt, long_filt2, short_filt, threshold, real_max_length,
                                           real_min_length, min_separation)

        # Initialize the filters. Create them default if None is passed
        if prefilter is None:
            wn = [100000, 150000]
            self.prefilter = Filter(filter_name='butter', order=4, frequencies=wn, filter_type='bandpass', fs=fs)
        else:
            self.prefilter = prefilter
            self.prefilter.fs = fs

        if dfilter is None:
            wn = 20000
            self.dfilter = Filter(filter_name='butter', order=2, frequencies=wn, filter_type='high', fs=fs)
        else:
            self.dfilter = dfilter
            self.dfilter.fs = fs

        self.save_max = save_max
        self.save_folder = save_folder
        self.save_noise = save_noise

        self.columns = ['id', 'datetime', 'filename', 'wave', 'start_sample', 'duration_samples',
                        'duration_us', 'amplitude']
        self.classifier = classifier
        if classifier is not None:
            if not self.check_classifier(classifier):
                raise TypeError('This classifier does not have a function to classify clicks!')
        if convert:
            if click_model_path is None:
                with resources.path('pyporcc.data', 'standard_click.wav') as click_model_path:
                    print('Setting the click model path to default...')
            self.converter = ClickConverter(click_model_path=click_model_path, fs=fs)
            self.columns += self.converter.click_vars
        else:
            self.converter = None

        self.saved_files = []

        # DataFrame with all the clips
        self.clips = pd.DataFrame(columns=self.columns)
        self.clips = self.clips.set_index('id')
        self.columns.remove('id')

        self.fs = fs

    def __setattr__(self, key, value):
        """
        If the sampling frequency of the sound is different than the one from the filters, update the filters 
        """
        if key == 'fs':
            self.dfilter.fs = value
            self.prefilter.fs = value
            if self.converter is not None:
                self.converter.fs = value
        elif key == 'save_folder':
            value = pathlib.Path(value)
        self.__dict__[key] = value

    @staticmethod
    def check_classifier(classifier):
        if hasattr(classifier, 'classify_click'):
            return True
        return False

    def pre_filter(self, xn):
        """
        Filter the sample xn with the pre-filter

        Parameters
        ----------
        xn : float
            Sample N
        """
        # Apply the prefilter to the sample
        xi = self.prefilter(xn)

        return xi

    def save_clips(self):
        """
        Save the clips in a file
        """
        # clips_filename_parquet = self.save_folder.joinpath('%s_clips.parquet.gzip' %
        #                                                 self.clips.datetime.iloc[0].strftime('%d%m%y_%H%M%S'))
        clips_filename_h5 = self.save_folder.joinpath('%s_clips.h5' %
                                                       self.clips.datetime.iloc[0].strftime('%d%m%y_%H%M%S'))
        waves_filename_h5 = self.save_folder.joinpath('%s_waves.h5' %
                                                       self.clips.datetime.iloc[0].strftime('%d%m%y_%H%M%S'))
        clips_filename_csv = self.save_folder.joinpath('%s_clips.csv' %
                                                       self.clips.datetime.iloc[0].strftime('%d%m%y_%H%M%S'))

        self.clips.filename = self.clips.filename.astype(str)
        self.clips.datetime = self.clips.datetime.astype(str)
        self.clips.start_sample = self.clips.start_sample.astype(int)
        # Save everything in the file
        if self.classifier is not None:
            if not self.save_noise:
                csv_df = self.clips.drop(index=self.clips.loc[self.clips[self.classifier.class_column] == 3].index)
            else:
                csv_df = self.clips
        else:
            csv_df = self.clips

        csv_df = csv_df.drop(columns=['wave'])
        csv_df.to_csv(clips_filename_csv)

        vlen_type = h5py.special_dtype(vlen=np.float32)
        f = h5py.File(waves_filename_h5, 'w')
        f.create_dataset('/waves', data=self.clips.wave.values, dtype=vlen_type)
        f.close()
        self.clips.drop(columns=['wave']).to_hdf(clips_filename_h5, key='clips', format='table')
        # self.clips.to_parquet(clips_filename_parquet, compression='gzip')
        self.clips.drop(index=self.clips.index, inplace=True)
        self.saved_files.append(clips_filename_h5)

    def add_click_clips(self, start_sample, blocksize, sound_file, clips_list):
        """
        Add all the clips list to the dictionary

        Parameters
        ----------
        start_sample : int
            Sample of the file where the block starts
        blocksize : int
            Number of frames to read in the block
        sound_file : SoundFile object
            File where the information is stored
        clips_list : list of tuples
            List containing all the clips of clicks as [start_sample, duration] in samples
        """
        sound_file.seek(start_sample)
        signal = sound_file.read(frames=blocksize)
        try:
            date = self.hydrophone.get_name_datetime(pathlib.Path(sound_file.name).name, utc=False)
        except:
            date = dt.datetime.now()
        date += dt.timedelta(seconds=start_sample / sound_file.samplerate)
        # Filter the signal
        filtered_signal = self.dfilter(signal)
        clips_block = self.clicks_block(filtered_signal, date, sound_file.name, start_sample, clips_list)
        self.clips = self.clips.append(clips_block, ignore_index=True, sort=False)

        # If longer than maximum, save it
        if len(self.clips) >= self.save_max:
            self.save_clips()

    def clicks_block(self, signal, date, filename, start_sample_block, clips_list, verbose=False):
        """
        Add the clip to the clip dictionary

        Parameters
        ----------
        signal : np.array
            Signal in the processed block
        date : datetime.datetime
            Datetime where the signal starts
        filename : string
            Name of the file where the signal is
        start_sample_block : int
            First sample of the block according to the whole file
        clips_list: list of tuples
            List containing all the clips of clicks as [start_sample, duration] in samples
        verbose : bool
            Set to True if plots are wanted
        """
        params_matrix = np.zeros((len(clips_list), len(self.columns)))
        timestamps = []
        waves = []
        signal_upa = utils.to_upa(signal, self.hydrophone.sensitivity, self.hydrophone.preamp_gain, self.hydrophone.Vpp)
        for idx, clip in enumerate(clips_list):
            start_sample = clip[0]
            duration_samples = clip[1]
            timestamp = date + dt.timedelta(seconds=start_sample / self.fs)
            # Read the clip
            istart = max(0, start_sample - self.pre_samples)
            frames = min(start_sample - istart + duration_samples + self.post_samples, signal.size)
            clip = signal_upa[istart:istart + frames]
            amplitude = utils.amplitude_db(clip)
            timestamps.append(timestamp)
            waves.append(clip)
            if self.converter is not None:
                q, duration, ratio, xc, cf, bw = self.converter.click_params(clip, nfft=512)
                params_matrix[idx, 3:len(self.columns)] = [start_sample_block + istart, frames, frames * 1e6 / self.fs,
                                                           amplitude, q, duration, ratio, xc, cf, bw]
            else:
                params_matrix[idx, 3:len(self.columns)] = [start_sample_block + istart, frames,
                                                           frames * 1e6 / self.fs, amplitude]
            if verbose:
                plt.figure(2, 1)
                plt.plot(clip, label='Filtered signal')
                plt.title('Signal filtered')
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.close()

        clips_block = pd.DataFrame(params_matrix, columns=self.columns)
        clips_block['wave'] = waves
        clips_block['datetime'] = timestamps
        clips_block['filename'] = filename
        clips_block.start_sample = clips_block.start_sample.astype(np.int32)

        if self.classifier is not None:
            clips_block = self.classifier.classify_matrix(clips_block)
        return clips_block

    def detect_click_clips_file(self, sound_file_path, blocksize=None):
        """
        Return the possible clips containing clicks

        Parameters
        ----------
        sound_file_path : string or Path
            Where the file to be computed is stored
        blocksize : int
            Number of samples to process at a time, if None it will be the length of the file.
            If the blocksize is too small (smaller than 1 period of the lowest frequency of the filter)
            it will affect the results
        """
        # Open file
        sound_file = sf.SoundFile(sound_file_path, 'r')

        # Update the frequency
        self.fs = sound_file.samplerate

        if blocksize is None:
            blocksize = sound_file.frames

        # Start the filter conditions
        zi = self.prefilter.get_zi()

        # Initialize the clicks count
        n_on = 0
        n_off = 0
        click_on = False

        # Read the file by blocks
        for block_n, block in enumerate(tqdm(sound_file.blocks(blocksize=blocksize, always_2d=True),
                                             total=sound_file.frames / blocksize, desc='file', leave=False)):
            prefilter_sig, zi = self.prefilter(block[:, 0], zi=zi)

            # Read samples one by one, apply filter
            clips, click_on, n_on, n_off = self.triggerfilter.update_block(prefilter_sig, click_on, n_on, n_off)
            self.add_click_clips(block_n * blocksize, blocksize, sound_file, clips)

        return self.clips

    def detect_click_clips_folder(self, folder_path, blocksize=None, zip_mode=False):
        """
        Go through all the sound files and create a database with the detected clicks.

        Parameters
        ----------
        folder_path : string or Path
            Where all the files are
        blocksize : int
            Number of samples to process at a time, if None it will be the length of the file.
            If the blocksize is too small (smaller than 1 period of the lowest frequency of the filter)
            it will affect the results
        zip_mode : bool
            Set to True if the files are zipped
        """
        if isinstance(folder_path, str):
            folder_path = pathlib.Path(folder_path)
        if zip_mode:
            folder_path = zipfile.ZipFile(folder_path, 'r', allowZip64=True)
            files_list = folder_path.namelist()
        else:
            files_list = sorted(folder_path.glob('*.wav'))
        for file_name in tqdm(files_list, total=len(files_list), desc='folder'):
            if file_name.suffix == '.wav':
                # Get the wav
                if zip_mode:
                    wav_file = folder_path.open(file_name)
                else:
                    wav_file = file_name
                self.detect_click_clips_file(wav_file, blocksize=blocksize)

        if self.save_max != np.inf:
            self.save_clips()

        return self.clips

    def classify_all_saved_clips(self, classifier=None):
        if classifier is not None:
            self.classifier = classifier
        else:
            if self.classifier is None:
                raise Exception('You need to provide one classifier in order to classify!')
        for f in self.saved_files:
            df = pd.read_pickle(f)
            df = self.classifier.classify_matrix(df)
            df.to_hdf5(f, mode='w', key='clips')


spec = [
    ('long_filt', nb.float32),
    ('long_filt2', nb.float32),
    ('short_filt', nb.float32),
    ('threshold', nb.int32),
    ('max_length', nb.int32),
    ('min_length', nb.int32),
    ('min_separation', nb.int32),
    ('Si', nb.float32),
    ('Ni', nb.float32),
    ('trigger', nb.boolean)
]


@nb.experimental.jitclass(spec)
class TriggerFilter:
    def __init__(self, long_filt, long_filt2, short_filt, threshold, max_length, min_length, min_separation):
        """
        Create a Trigger Filter
        The noise level N at sample i is measured using
        Ni = alpha_n * |x|i + (1 - alpha_n) * Ni_1
        and the signal level S is measured using
        Si = alpha_s * |x|i + (1 - alpha_s) * Si_1
        where N is either the long filter parameter when no click is active (i.e. the signal is below threshold)
        or the Long Filter 2 parameter when the signal is above threshold. S is the Short filter parameter.
        A click is started / stopped when the ratio S/N goes above / below the Threshold parameter.

        Parameters
        ----------
        long_filt : float
            Long filter (used when no click is active)
        long_filt2 : float
            Long filter 2 (used when signal is above threshold)
        short_filt: float
            Short filter parameter
        threshold : float
            Detection threshold in db
        min_separation : int
            Minimum separation between clicks in samples
        min_length : int
            Minimum length of a click in samples
        max_length : int
            Maximum length of a click in samples
        """
        self.long_filt = long_filt
        self.long_filt2 = long_filt2
        self.short_filt = short_filt
        self.threshold = 10 ** (threshold / 20.0)
        self.max_length = max_length
        self.min_length = min_length
        self.min_separation = min_separation

        self.Si = 0
        self.Ni = 0

        self.trigger = False

    def initialize_sn(self, block):
        """
        Initialize the values of Si and Ni for correct detection

        Parameters
        ----------
        block : np.array
            Signal to initialize the filter with
        """
        self.Si = np.abs(block).mean() * self.threshold
        self.Ni = np.abs(block).mean()

    def run(self, xi):
        """
        Raise the trigger if SNR is higher than the threshold

        Parameters
        ----------
        xi : float
            Sample i
        """
        # Calculate the SNR 
        self.Si = self.short_filt * np.abs(xi) + (1 - self.short_filt) * self.Si

        if self.trigger:
            self.Ni = self.long_filt2 * np.abs(xi) + (1 - self.long_filt2) * self.Ni
        else:
            self.Ni = self.long_filt * np.abs(xi) + (1 - self.long_filt) * self.Ni

        # Compute SNR = Si/Ni and compare it to the threshold
        if (self.Si / self.Ni) > self.threshold:
            self.trigger = True
        else:
            self.trigger = False

    def update_block(self, prefilter_signal, click_on, n_on, n_off):
        """
        Read all the samples of the block and update the parameters

        Parameters
        ----------
        prefilter_signal : np.array
            Signal to process, already filtered
        click_on : bool
            Set to True if the click was on on the last sample
        n_on : int
            Number of samples the click has been on
        n_off : int
            Number of samples the click has been off
        """
        if click_on:
            start_sample = -n_on
        else:
            start_sample = 0
        clips = []
        i = 0
        for xi in prefilter_signal:
            self.run(xi)
            # If it has been on for too long, save the click!
            if click_on:
                if self.trigger:
                    # Continue the click
                    if n_off > 0:
                        # If it is triggered but the sum of n_on and n_off is already too much
                        # We can't consider it the same click. In case n_on was long enough it is saved
                        # Otherwise it is just discarded
                        if n_on + n_off >= self.max_length:
                            if n_on >= self.min_length:
                                clips.append((start_sample, n_on))
                            click_on = False
                            n_on = 0
                            n_off = 0
                        else:
                            # In case we were already in the count down but it is actually the same click,
                            # consider the count down as part of the click!
                            n_on += n_off
                            n_off = 0
                    n_on += 1
                else:
                    # If it has been off for more than min_separation, save the click! 
                    if n_off >= self.min_separation:
                        if n_on >= self.min_length:
                            clips.append((start_sample, n_on))
                            click_on = False
                            n_on = 0
                            n_off = 0
                    else:
                        # Start to end the click
                        n_off += 1
            else:
                if self.trigger:
                    # Start a new click 
                    start_sample = i
                    click_on = True
                    n_on = 0
                    n_off = 0
            if n_on >= self.max_length:
                clips.append((start_sample, n_on))
                click_on = False
                n_on = 0
                n_off = 0
            i += 1

        return clips, click_on, n_on, n_off


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
    x_coeff = np.correlate(sound_block, click_model)
    xc = np.max(x_coeff)

    return q, duration, ratio, xc, cf, bw, psd, freq


class ClickConverter:
    def __init__(self, click_model_path, fs, click_vars=None):
        """
        Init the click converter

        Parameters
        ----------
        click_model_path : string or Path
            File where the click model is stored
        click_vars : list of strings
            List of the output parameters to compute for each click
        """
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
            params_matrix[idx, :] = self.click_params(row['wave'], nfft=512)

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


class Filter:
    def __init__(self, filter_name, filter_type, frequencies, order, fs=None):
        """
        Filter object that allows to change the sampling frequency keeping the other parameters
        """
        if filter_name not in ['butter', 'cheby1', 'cheby2', 'besel']:
            filter_name = 'butter'
            raise Warning('Filter %s is not implemented or unknown, setting to Butterworth' % filter_name)
        self.filter_name = filter_name
        self.filter_type = filter_type
        self.frequencies = frequencies
        self.order = order
        self.fs = fs
        if fs is not None:
            self.get_filt(fs)
        else:
            self.filter = None

    def __setattr__(self, key, value):
        if key == 'fs':
            if value is not None:
                self.get_filt(value)
        self.__dict__[key] = value

    def __call__(self, signal, zi=None):
        if zi is None:
            return sig.sosfilt(self.filter, signal)
        else:
            return sig.sosfilt(self.filter, signal, zi=zi)

    def get_filt(self, fs):
        filt = getattr(sig, self.filter_name)
        if type(self.frequencies) is list:
            wn = [f / (fs / 2.0) for f in self.frequencies]
            self.filter = filt(N=self.order, Wn=wn, btype=self.filter_type, analog=False, output='sos')
        else:
            self.filter = filt(N=self.order, Wn=self.frequencies, btype=self.filter_type, analog=False,
                               output='sos', fs=fs)

    def get_zi(self):
        return sig.sosfilt_zi(self.filter)
