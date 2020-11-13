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

from pyporcc import utils

import os
import zipfile
import pathlib
import numpy as np
import numba as nb
import pandas as pd
from tqdm import tqdm
import datetime as dt
import soundfile as sf
from importlib import resources
import matplotlib.pyplot as plt 

from scipy import signal as sig
from scipy import interpolate


class ClickDetector:
    def __init__(self, hydrophone=None, long_filt=0.00001, long_filt2=0.000001, short_filt=0.1, threshold=10,
                 min_separation=100, max_length=1024, min_length=100, pre_samples=40, post_samples=40, fs=576000,
                 prefilter=None, dfilter=None, save_max=np.inf, save_folder='.',
                 convert=False, click_model_path=None, classifier=None):
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
        """
        self.hydrophone = hydrophone
        # Detector parameters
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.fs = fs
        
        self.triggerfilter = TriggerFilter(long_filt, long_filt2, short_filt, threshold, max_length,
                                           min_length, min_separation)

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

        # DataFrame with all the clips
        self.clips = pd.DataFrame(columns=['id', 'datetime', 'start_sample', 'duration_samples',
                                           'duration_us', 'amplitude', 'filename', 'wave'])
        self.clips = self.clips.set_index('id')
        self.save_max = save_max
        self.save_folder = pathlib.Path(save_folder)

        self.classifier = classifier
        if classifier is not None:
            if not self.check_classifier(classifier):
                raise TypeError('This classifier does not have a function to classify clicks!')
            # self.classifier.class_column = 'class_type'
        if convert:
            if click_model_path is None:
                with resources.path('pyporcc.data', 'standard_click.wav') as click_model_path:
                    print('Setting the click model path to default...')
            self.converter = ClickConverter(click_model_path=click_model_path)
            for var in self.converter.click_vars:
                self.clips[var] = None
        else:
            self.converter = None

        self.saved_files = []

    def __setitem__(self, key, value):
        """
        If the sampling frequency of the sound is different than the one from the filters, update the filters 
        """
        if key == 'fs':
            self.dfilter.fs = value
            self.prefilter.fs = value
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
        clips_filename_pkl = self.save_folder.joinpath('Detected_Clips_%s.pkl' %
                                                       self.clips.iloc[0].datetime.strftime('%d%m%y_%H%M%S'))
        clips_filename_csv = self.save_folder.joinpath('Detected_Clicks_%s.csv' %
                                                       self.clips.iloc[0].datetime.strftime('%d%m%y_%H%M%S'))
        self.clips.to_pickle(clips_filename_pkl)
        # clicks_mask = (self.clips.class_type == 1) | (self.clips.class_type == 2)
        # self.clips[clicks_mask].drop(columns=['wave']).to_csv(clips_filename_csv)
        self.clips.drop(index=self.clips.index, inplace=True)
        self.saved_files.append(clips_filename_pkl)

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
        date = self.hydrophone.get_name_datetime(pathlib.Path(sound_file.name).name)
        date += dt.timedelta(seconds=start_sample/sound_file.samplerate)
        # Filter the signal
        filtered_signal = self.dfilter(signal)
        for clip in clips_list:
            self.add_click_clip(filtered_signal, sound_file, date, start_sample, clip[0], clip[1])

    def add_click_clip(self, signal, sound_file, date, start_sample_block, start_sample,
                       duration_samples=0, verbose=False):
        """
        Add the clip to the clip dictionary

        Parameters
        ----------
        signal : np.array
            Signal in the processed block
        sound_file : SoundFile object
            Sound file where the clip is stored
        date : datetime.datetime
            Datetime where the signal starts
        start_sample_block : int
            First sample of the block according to the whole file
        start_sample : int
            Number of sample of the first sample of the click
        duration_samples : int
            Duration of the click in samples
        verbose : bool
            Set to True if plots are wanted
        """
        timestamp = date + dt.timedelta(seconds=start_sample/sound_file.samplerate)

        # Read the clip 
        istart = max(0, start_sample - self.pre_samples)
        frames = min(start_sample - istart + duration_samples + self.post_samples, signal.size)
        clip = signal[istart:istart+frames]

        amplitude = utils.amplitude_db(clip, self.hydrophone.sensitivity, self.hydrophone.preamp_gain,
                                       self.hydrophone.Vpp)
        id = len(self.clips)
        self.clips.at[id, ['datetime', 'start_sample', 'wave', 'duration_samples', 'duration_us',
                           'amplitude', 'filename']] = [timestamp, start_sample_block + start_sample, clip,
                                                        duration_samples, duration_samples*1e6/self.fs,
                                                        amplitude, pathlib.Path(sound_file.name).name]
        if self.converter is not None:
            self.clips.at[id] = self.converter.convert_row(self.clips.iloc[id], fs=sound_file.samplerate).values
        # if self.classifier is not None:
        #     self.clips.at[id, 'class_type'] = self.classifier.classify_row(self.clips.iloc[id])
        if verbose:
            fig, ax = plt.subplots(2, 1)
            # ax[0].plot(clip, label='Signal not filtered')
            ax[1].plot(clip, label='Filtered signal')
            # ax[0].set_title('Signal not filtered')
            ax[1].set_title('Signal filtered')
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()

        if len(self.clips) >= self.save_max:
            if self.classifier is not None:
                self.clips = self.classifier.classify_matrix(self.clips)
            self.save_clips()

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
                                             total=sound_file.frames / blocksize)):
            prefilter_sig, zi = self.prefilter(block[:, 0], zi=zi)

            # Read samples one by one, apply filter
            clips, click_on, n_on, n_off = self.triggerfilter.update_block(prefilter_sig, click_on, n_on, n_off)
            self.add_click_clips(block_n*blocksize, blocksize, sound_file, clips)
        # Save the last detected clips
        if self.save_max != np.inf:
            self.save_clips()

        return self.clips

    def get_click_clips_folder(self, folder_path, zip_mode=False):
        """
        Go through all the sound files and create a database with the detected clicks.

        Parameters
        ----------
        folder_path : string or Path
            Where all the files are
        zip_mode : bool
            Set to True if the files are zipped
        """
        # Run the classifier in every file of the folder
        for day_folder_name in os.listdir(folder_path):
            day_folder_path = os.path.join(folder_path, day_folder_name)
            if zip_mode:
                day_folder_path = zipfile.ZipFile(day_folder_path, 'r', allowZip64=True)
                files_list = day_folder_path.namelist()
            else:
                files_list = os.listdir(day_folder_path)
            for file_name in files_list:
                extension = file_name.split(".")[-1]
                if extension == 'wav':
                    print(file_name)
                    # Get the wav
                    if zip_mode:
                        wav_file = day_folder_path.open(file_name)
                    else:
                        wav_file = os.path.join(day_folder_path, file_name)
                    self.detect_click_clips_file(wav_file)
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
            df.to_pickle(f)



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
            if click_on:
                if self.trigger: 
                    # Continue the click 
                    # In case we were already in the count down but it is actually the same click,
                    # consider the count down as part of the click!
                    if n_off > 0: 
                        n_on += n_off
                        n_off = 0
                    n_on += 1

                    # If it has been on for too long, save the click! 
                    if n_on >= self.max_length:
                        clips.append((start_sample, n_on))
                        click_on = False
                        n_on = 0
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
            
            i += 1

        return clips, click_on, n_on, n_off


class Click:
    def __init__(self, sound_block, fs, timestamp, click_model_path, nfft=64, verbose=False):
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
            new_samples = int(np.ceil(self.click_model.size/fs_model * self.fs))
            self.click_model = sig.resample(self.click_model, new_samples)
        else:
            self.fs = fs

        # [PSD,f] = periodogram(click,[],FFT,Fs,'power');
        # %psd=abs(fft(click))
        # PSD = PSD./max(PSD); % normalize spectrum
        # CF = sum(f.*PSD.^2)./sum(PSD.^2);
        # [~,indfc] = max(PSD);
        # PF = f(indfc); 

        # Calculate PSD, freq, centrum-freq (cf), peak-freq (pf) of the sound file 
        # window = sig.get_window('boxcar', self.nfft)
        window = sig.get_window('boxcar', self.nfft)
        self.freq, psd = sig.periodogram(x=sound_block, window=window, nfft=self.nfft, fs=self.fs, scaling='spectrum')

        # Normalize spectrum
        self.psd = psd / np.max(psd)
        self.cf = np.sum(self.freq * (psd**2)) / np.sum(psd**2)
        self.pf = self.freq[psd.argmax()]    

        # Calculate RMSBW
        # RMSBW = (sqrt(sum((f-CF).^2.*PSD.^2 ) / sum(PSD.^2)))/1000;
        self.rmsbw = (np.sqrt((np.sum(((self.freq - self.cf)**2) * (self.psd**2))) / np.sum(self.psd**2))) / 1000.0 

        # Calculate click duration based on Madsen & Walhberg 2007 - 80#
        ener = np.cumsum(self.sound_block**2)
        istart = np.where(ener <= (ener[-1] * 0.1))[0]          # index of where the 1.5% is
        iend = np.where(ener <= (ener[-1] * 0.9))[0]            # index of where the 98.5% is
        if len(istart) > 0:
            istart = istart[-1]
        else:
            istart = 0
        if len(iend) > 0: 
            iend = iend[-1]
        else:
            iend = len(ener)
        self.duration = (iend - istart) / self.fs * 1e6         # duration in microseconds
                
        # Parameters according to Mhl & Andersen, 1973 
        self.q = (self.cf / self.rmsbw) / 1000.0
        self.ratio = self.pf / self.cf
        
        # Calculate -3dB bandwith: Consecutive frequencies of the psd that have more than half of the maximum freq power
        half = np.max(psd) / (10 ** (3/10.0))
        max_freq_i = psd.argmax()
        i = np.where(psd[0:max_freq_i] <= half)[0][-1]
        inter = interpolate.interp1d(psd[i: i + 2], self.freq[i: i + 2])
        f_left = inter(half)

        i = np.where(psd[max_freq_i+1:-1] <= half)[0][0] + max_freq_i + 1
        inter = interpolate.interp1d(psd[i - 1: i + 1], self.freq[i - 1: i + 1])
        f_right = inter(half)

        self.bw = (f_right - f_left)/1000.0

        # Calculate the correlation with the model
        x_coeff = np.correlate(self.sound_block, self.click_model, mode='same')
        self.xc = np.max(x_coeff)

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


class ClickConverter:
    def __init__(self, click_model_path, click_vars=None):
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
        if click_vars is None:
            self.click_vars = ['Q', 'duration', 'ratio', 'XC', 'CF', 'BW']
        else:
            self.click_vars = click_vars

    def clicks_df(self, df, save_path=None):
        """
        Find all the clicks and return them as a df with all the clicks params

        Parameters
        ----------
        df : DataFrame
            It needs to have at least datetime and wave
        save_path : string or Path
            Path to the desired save file. If None, it is not saved
        """
        # Calculate all the independent variables and append them to the df
        for var in self.click_vars:
            df[var] = 0.00

        for idx in df.index:
            signal = df.loc[idx, 'wave']
            click = Click(signal, df.fs, df.loc[idx, 'datetime'], click_model_path=self.click_model_path, verbose=False)
            values = []
            for var in self.click_vars:
                values.append(getattr(click, var))
            df.loc[idx, self.click_vars] = values
        
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
            else:
                raise Exception('The extension %s is unkown' % extension)

        return df

    def convert_row(self, row, fs):
        row = row.copy()
        signal = row['wave']
        if 'datetime' in row.axes[0]:
            dt = row.datetime
        else:
            dt = row.name
        click = Click(signal, fs, dt, click_model_path=self.click_model_path, verbose=False)
        for var in self.click_vars:
            row[var] = getattr(click, var)

        return row

    @staticmethod
    def test_click_calculation(df_clicks, df_test, col_vars):
        """
        Test the calculation of the click parameters indicated in col_vars obtained with python
        compared to the ones obtained in the paper (on the Test DB)
        """
        # Compare each field
        rel_error = np.abs(df_test[col_vars] - df_clicks[col_vars])/df_test[col_vars].mean()
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
            wn = [f/(fs/2.0) for f in self.frequencies]
            self.filter = filt(N=self.order, Wn=wn, btype=self.filter_type, analog=False, output='sos')
        else:
            self.filter = filt(N=self.order, Wn=self.frequencies, btype=self.filter_type, analog=False,
                               output='sos', fs=fs)

    def get_zi(self):
        return sig.sosfilt_zi(self.filter)
