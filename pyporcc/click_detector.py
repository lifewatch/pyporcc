import math 
import numpy as np
import matplotlib.pyplot as plt 
import os
import configparser
import datetime as dt
from scipy.io import wavfile as siowavfile
from scipy import signal as sig
import soundfile as sf
import pandas as pd
import numba as nb




class ClickDetector:
    def __init__(self, long_filt=0.00001, long_filt2=0.000001, short_filt=0.1, threshold=10, min_separation=100, max_length=1024, pre_samples=40, post_samples=40, fs=500000, prefilter=None, dfilter=None):
        """
        Process to detect clicks
        """
        # Detector parameters
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.fs = fs
        
        self.triggerfilter = TriggerFilter(long_filt, long_filt2, short_filt, threshold, max_length, min_separation)

        # Initialize the filters. Create them default if None is passed
        if prefilter is None:
            self.prefilter = sig.butter(N=4, Wn=500, btype='high', analog=False, output='sos', fs=fs)
        else:
            self.prefilter = prefilter
        
        if dfilter is None:
            self.dfilter = sig.butter(N=4, Wn=2000, btype='high', analog=False, output='sos', fs=fs)
        else:
            self.dfilter = dfilter

        # Parameters for recursive reading
        self.filter_state = np.zeros((2,2))

        # Dictionary with all the clips
        self.clips = {}


    def __setitem__(self, key, value):
        """
        If the sampling frequency of the sound is different than the one from the filters, update the filters 
        """
        if key == 'fs':
            if self.__dict__[key] != value: 
                self.prefilter = sig.butter(N=4, Wn=500, btype='high', analog=False, output='sos', fs=value)
                self.dfilter = sig.butter(N=4, Wn=2000, btype='high', analog=False, output='sos', fs=value)   
                self.__dict__[key] = value             
                

    def pre_filter(self, xn):
        """
        Filter the sample xi with the pre-filter
        """
        # Apply the prefilter to the sample
        xi = sig.sosfilt(self.prefilter, xn)

        return xi


    def add_click_clips(self, sound_file, date, block_n, clips_list):
        """
        Add all the clips list to the dictionary 
        """
        for clip in clips_list:
            self.add_click_clip(sound_file, date, block_n*clip[0], block_n*clip[1])



    def add_click_clip(self, sound_file, date, start_sample, frames):
        """
        Add the clip to the clip dictionary 
        """
        timestamp = date + dt.timedelta(seconds=start_sample/sound_file.samplerate)

        # Read the clip 
        istart = max(0, start_sample - self.pre_samples)
        frames = min(start_sample - istart + self.post_samples, sound_file._info.frames)
        sound_file.seek(istart)
        clip = sound_file.read(frames=frames, always_2d=True)[:,0]
        
        # Filter the clip and add it to the dictionary
        filtered_clip = sig.sosfilt(self.dfilter, clip)
        self.clips[timestamp] = filtered_clip


    def clip2click(self, datetime, signal, click_model_path):
        """
        Return a click object from a 
        """
        clip = self.clips[datetime]
        click = Click(signal, self.fs, datetime, click_model_path=click_model_path)

        return click 


    def detect_click_clips(self, sound_file_path, date, blocksize=2048):
        """
        Return the possible clips containing clicks
        """
        # Open file
        sound_file = sf.SoundFile(sound_file_path, 'r')

        # Initialize the values of Si and Ni using the first block
        first_block = sound_file.read(frames=blocksize, always_2d=True)[:,0]
        sound_file.seek(blocksize)
        self.triggerfilter.initialize_SN(first_block)

        # Initialize the clicks count
        n_on = 0
        n_off = 0
        click_on = False

        # Read the file by blocks
        for block_n, block in enumerate(sound_file.blocks(blocksize=blocksize, always_2d=True)):
            prefilter_sig = sig.sosfilt(self.prefilter, block[:,0])
            # Read samples one by one, apply filter
            clips, click_on, n_on, n_off = self.triggerfilter.update_block(prefilter_sig, click_on, n_on, n_off)
            self.add_click_clips(sound_file, date, block_n, clips)
             


    def clicks_df(self, click_model_path):
        """
        Find all the clicks and return them as a df with all the clicks params
        """
        cols=['datetime', 'Q', 'duration', 'ratio', 'XC', 'CF', 'BW']
        clicks_df = pd.DataFrame(columns=cols)

        idx = 0
        for datetime, wave in self.clips.items():
            click = self.clip2click(datetime, wave, click_model_path)
            clicks_df.loc[idx, cols] = [click.timestamp, click.Q, click.duration, click.ratio, click.xc, click.cf, click.bw]
            idx += 1

        return clicks_df




spec = [
    ('long_filt', nb.float32),
    ('long_filt2', nb.float32),
    ('short_filt', nb.float32),
    ('threshold', nb.int32),
    ('max_length', nb.int32),
    ('min_separation', nb.int32),
    ('Si', nb.float32),
    ('Ni', nb.float32),
]

@nb.jitclass(spec)
class TriggerFilter:
    def __init__(self, long_filt, long_filt2, short_filt, threshold, max_length, min_separation):
        """
        Create a Trigger Filter using jitclass to make it faaaaaaast
        """
        self.long_filt = long_filt
        self.long_filt2 = long_filt2
        self.short_filt = short_filt
        self.threshold = 10 ** (threshold / 20.0)
        self.max_length = max_length
        self.min_separation = min_separation

        self.Si = 0
        self.Ni = 0


    def initialize_SN(self, block):
        """
        Initialize the values of Si and Ni for correct detection 
        """
        self.Si = np.abs(block).mean() * self.threshold
        self.Ni = np.abs(block).mean()

    
    def run(self, xi, click_on):
        """
        Raise the trigger if SNR is higher than the threshold
        """
        # Calculate the SNR 
        self.Si = self.short_filt * np.abs(xi) + (1 - self.short_filt) * self.Si

        if click_on:
            self.Ni = self.long_filt2 * np.abs(xi) + (1 - self.long_filt2) * self.Ni
        else: 
            self.Ni = self.long_filt * np.abs(xi) + (1 - self.long_filt) * self.Ni
        
        # Compute SNR = Si/Ni and compare it to the threshold
        if (self.Si / self.Ni) > self.threshold:
            return True
        else:
            return False


    def update_block(self, prefilter_signal, click_on, n_on, n_off):
        """
        Read all the samples of the blog and update the parameters
        """
        clips = []
        i = 0
        for xi in prefilter_signal:    
            if self.run(xi, click_on):
                if click_on:
                    if n_on == self.max_length:
                        clips.append((i, n_on))
                        n_on = 0
                        click_on = False
                else:
                    click_on = True
                n_on += 1
                    
            else:
                if click_on:
                    if n_off >= self.min_separation:
                        # self.add_click_clip(sound_file, date, i*block_n, n_on) 
                        clips.append((i, n_on))
                        n_on = 0 
                        n_off = 0 
                        click_on = False 
                        n_off += 1  
            i += 1

        return clips, click_on, n_on, n_off

 

class Click:
    def __init__(self, sound_block, fs, timestamp, click_model_path, nfft=512, verbose=False):
        """
        Start a porpoise click
        """
        self.nfft = nfft
        self.fs = fs
        self.sound_block = sound_block
        self.timestamp = timestamp
        fs_model, self.click_model = siowavfile.read(click_model_path)
        if fs_model != fs: 
            if verbose:
                print('This click is not recorded at the same frequency than the classified data! Resampling to %s S/s' % (self.fs))
            new_samples = int(np.ceil(self.click_model.size/fs_model * self.fs))
            self.click_model = sig.resample(self.click_model, new_samples)
        else:
            self.fs = fs

        # Calculate PSD, freq, centrum-freq (cf), peak-freq (pf) of the sound file 
        # window = sig.get_window('boxcar', self.nfft)
        window = sig.get_window('boxcar', 0)
        self.freq, psd = sig.periodogram(x=sound_block, fs=self.fs, nfft=self.nfft, scaling='spectrum')

        # Normalize spectrum
        self.psd = psd / psd.max()
        self.cf = np.sum(self.freq * (psd**2)) / np.sum(psd**2)
        self.pf = self.freq[psd.argmax()]    

        # Calculate RMSBW
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
        self.Q = (self.cf / self.rmsbw) / 1000.0
        self.ratio = self.pf / self.cf
        
        # Calculate -3dB bandwith: Consecutive frequencies of the psd that have more than half of the maximum freq power
        half = psd.max() / (10 ** (3/10.0))
        max_freq_i = psd.argmax()
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



def read_calibration(serial_number):
    """
    Read the calibration file of the sountrap serial number and store it in a dictionary
    """
    configfile_path = os.path.join('pyporcc/calibration', str(serial_number)+'.ini')
    config = configparser.ConfigParser()
    config.read(configfile_path)
    high_gain = float(config["End-to-End_Calibration"]["High_Gain_dB"])
    low_gain = float(config["End-to-End_Calibration"]["Low_Gain_dB"])
    rti_level = float(config["Calibration_Tone"]["RTI_Level_at_1kHz_dB_re_1_uPa"])
    calibration = {"High": high_gain, "Low": low_gain, "RTI": rti_level}

    return calibration


def wav2uPa(calibration, signal):
    """
    Convert the values of the wav file to the levels of SPL according to the calibration
    """
    # Calibration value from dB to ratio
    cal_value = np.power(10, calibration / 20.0)
    signal_uPa = signal * cal_value

    return signal_uPa


def uPa2spl(samples):
    """
    Convert from uPa to dB
    """
    spl = 20 * np.log10(samples)

    return spl
