#!/usr/bin/python
"""
Module : click_detector.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Instituut voor de Zee)
Last Accessed : 9/23/2020
"""

import os
import zipfile
import numpy as np
import numba as nb
import datetime as dt
import soundfile as sf
import matplotlib.pyplot as plt 

from scipy.io import wavfile as siowavfile
from scipy import signal as sig


class ClickDetector:
    def __init__(self, long_filt=0.00001, long_filt2=0.000001, short_filt=0.1, threshold=10, min_separation=100,
                 max_length=1024, pre_samples=40, post_samples=40, fs=500000, prefilter=None, dfilter=None):
        """
        Process to detect clicks
        Trigger decision
        The trigger automatically makes a measure of background noise and then compares the signal level to the
        noise level. When the signal level reaches a certain threshold above the noise level a click clip is started.
        When the signal level falls to below the threshold for more than a set number of bins, the click clip ends and
        the clip is sent to the localisation and classification modules.
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
        prefilter: filter output from scipy
            Prefilter to apply to the signal before passing it to the trigger. It is also applied to the stored clips.
            If set to None a band-pass Butterworth 4th order filter [100000, 150000] Hz will be used
        dfilter : filter output from scipy
            Filter to apply for the trigger.
            If set to None, a high-pass Butterworth [20000, :] 4th order filter will be used
        """
        # Detector parameters
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.fs = fs
        
        self.triggerfilter = TriggerFilter(long_filt, long_filt2, short_filt, threshold, max_length, min_separation)

        # Initialize the filters. Create them default if None is passed
        if prefilter is None:
            nyq = fs/2
            wn = [100000/nyq, 150000/nyq] 
            self.prefilter = sig.butter(N=4, Wn=wn, btype='bandpass', analog=False, output='sos')
        else:
            self.prefilter = prefilter
        
        if dfilter is None:
            wn = 20000
            self.dfilter = sig.butter(N=2, Wn=wn, btype='high', analog=False, output='sos', fs=fs)
        else:
            self.dfilter = dfilter

        # Dictionary with all the clips
        self.clips = {}

    def __setitem__(self, key, value):
        """
        If the sampling frequency of the sound is different than the one from the filters, update the filters 
        """
        if key == 'fs':
            if self.__dict__[key] != value: 
                wn_pre = [100000, 150000] 
                self.prefilter = sig.butter(N=8, Wn=wn_pre, btype='bandpass', analog=False, output='sos', fs=value)
                wn_d = 20000
                self.dfilter = sig.butter(N=4, Wn=wn_d, btype='high', analog=False, output='sos', fs=value)
        self.__dict__[key] = value         

    def pre_filter(self, xn):
        """
        Filter the sample xn with the pre-filter

        Parameters
        ----------
        xn : float
            Sample N
        """
        # Apply the prefilter to the sample
        xi = sig.sosfilt(self.prefilter, xn)

        return xi

    def add_click_clips(self, sound_file, date, block_start_sample, clips_list):
        """
        Add all the clips list to the dictionary

        Parameters
        ----------
        sound_file : SoundFile object
            Where the file to be computed is stored
        date : datetime
            Start datetime of the file
        block_start_sample : int
            Number of sample of the first sample of the block
        clips_list : list of tuples
            List containing all the clips of clicks as [start_sample, duration] in samples
        """
        for clip in clips_list:
            self.add_click_clip(sound_file, date, block_start_sample+clip[0], clip[1])

    def add_click_clip(self, sound_file, date, start_sample, duration_samples=0, verbose=False):
        """
        Add the clip to the clip dictionary

        Parameters
        ----------
        sound_file : SoundFile object
            Where the file to be computed is stored
        date : datetime
            Start datetime of the file
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
        frames = min(start_sample - istart + duration_samples + self.post_samples, sound_file.frames)
        sound_file.seek(istart)
        clip = sound_file.read(frames=frames, always_2d=True)[:, 0]
        
        # Filter the clip and add it to the dictionary
        filtered_clip = sig.sosfilt(self.dfilter, clip)
        self.clips[timestamp] = (start_sample, filtered_clip)

        if verbose:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(clip, label='Signal not filtered')
            ax[1].plot(filtered_clip, label='Filtered signal')
            plt.tight_layout()
            plt.show()
            plt.close()

    def detect_click_clips_file(self, sound_file_path, date, blocksize=2048):
        """
        Return the possible clips containing clicks

        Parameters
        ----------
        sound_file_path : string or Path
            Where the file to be computed is stored
        date : datetime
            Start datetime of the file
        blocksize : int
            Number of samples to process at a time
        """
        # Open file
        sound_file = sf.SoundFile(sound_file_path, 'r')

        # Update the frequency 
        if sound_file.samplerate != self.fs: 
            self.fs = sound_file.samplerate

        # Start the filter conditions
        zi = sig.sosfilt_zi(self.prefilter)

        # Initialize the values of Si and Ni using the first block
        first_block = sound_file.read(frames=blocksize, always_2d=True)[:, 0]
        zi = first_block.mean() * zi
        filtered_first_block, zi = sig.sosfilt(self.prefilter, first_block, zi=zi)
        zi = first_block.mean() * zi
        self.triggerfilter.initialize_sn(filtered_first_block)

        # Initialize the clicks count
        n_on = 0
        n_off = 0
        click_on = False

        # Read the file by blocks
        for block_n, block in enumerate(sound_file.blocks(blocksize=blocksize, always_2d=True)):
            prefilter_sig, zi = sig.sosfilt(self.prefilter, block[:, 0], zi=zi)

            # Read samples one by one, apply filter
            clips, click_on, n_on, n_off = self.triggerfilter.update_block(prefilter_sig, click_on, n_on, n_off)
            self.add_click_clips(sound_file, date, block_n*blocksize, clips)

            # plt.plot(prefilter_sig, label='Filtered')
            # plt.show()

    def get_click_clips(self, hydrophone, folder_path, zip_mode=False):
        """
        Go through all the sound files and create a database with the detected clicks.

        Parameters
        ----------
        hydrophone : hydrophone object
            Object representing a hydrophone from pyhydrophone
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
                    date = hydrophone.get_name_datetime(file_name)
                    self.detect_click_clips_file(wav_file, date)

        return self.clips


spec = [
    ('long_filt', nb.float32),
    ('long_filt2', nb.float32),
    ('short_filt', nb.float32),
    ('threshold', nb.int32),
    ('max_length', nb.int32),
    ('min_separation', nb.int32),
    ('Si', nb.float32),
    ('Ni', nb.float32),
    ('trigger', nb.boolean)
]


@nb.jitclass(spec)
class TriggerFilter:
    def __init__(self, long_filt, long_filt2, short_filt, threshold, max_length, min_separation):
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
        max_length : int
            Maximum length of a click in samples
        """
        self.long_filt = long_filt
        self.long_filt2 = long_filt2
        self.short_filt = short_filt
        self.threshold = 10 ** (threshold / 20.0)
        self.max_length = max_length
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
        fs_model, self.click_model = siowavfile.read(click_model_path)
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
        self.freq, psd = sig.periodogram(x=sound_block, window=window, fs=self.fs, scaling='spectrum')

        # Normalize spectrum
        self.psd = psd / psd.max()
        self.cf = np.sum(self.freq * (psd**2)) / np.sum(psd**2)
        self.pf = self.freq[psd.argmax()]    

        # Calculate RMSBW
        # BW = (sqrt(sum((f-CF).^2.*PSD.^2 ) / sum(PSD.^2)))/1000; 
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
        half = psd.max() / (10 ** (3/10.0))
        max_freq_i = psd.argmax()
        i_left, i_right = 0, 0
        for i in np.arange(0, max_freq_i):
            if psd[max_freq_i - i] < half:
                break
            else:
                i_left = max_freq_i - i

        for i in np.arange(0, psd.size - max_freq_i):
            if psd[max_freq_i + i] < half:
                break
            else:
                i_right = max_freq_i + i

        self.bw = (self.freq[i_right] - self.freq[i_left])/1000.0

        # Calculate the correlation with the model
        x_coeff = np.correlate(self.sound_block, self.click_model, mode='same')
        self.xc = x_coeff.max()

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
