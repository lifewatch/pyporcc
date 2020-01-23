import math 
import numpy as np
import matplotlib.pyplot as plt 
import os
import configparser
import datetime as dt
from scipy import signal as sig
from scipy.io import wavfile
import soundfile as sf

import xml.etree.ElementTree as ET



class SoundTrapFile:
    def __init__(self, wavfile, xmlfile, calibration, last_gain='HIGH', max_num_samples=None):
        """
        Data recorded in a wav file. Return the type 'NEW' or 'REOPEN' to use the config of last file
        """
        # Get the date from the name (XML data is not working!!)
        self.date = self.get_name_info(wavfile.name)

        self.wavfile = wavfile
        self.xmlfile = xmlfile

        # Read the specs, set the gain. Will read the fs as well (the good one, not zipped)
        self.read_file_specs(xmlfile, last_gain)

        # According to the specs set the calibration number
        self.calibration = calibration[self.st_gain]

        self.sound_file = sf.SoundFile(self.wavfile, 'r')

        # Max number of samples to read is a multiple of 2 to the power (close to 3 min)
        self.max_num_samples = int(2 ** np.ceil(np.log2(1*60*self.fs)))


    def read_file_specs(self, xmlfile_path, last_gain):
        """
        Read the specs of the recording from the XML file and save them to the object
        """
        tree = ET.parse(xmlfile_path)
        self.type_start = tree.find('EVENT/START').get('STATE')

        # Metadata colected 
        self.temp = float(tree.find('EVENT/TEMPERATURE').text)/100

        # WavFileHandler information
        self.sampling_attr = {}
        WavFileHandler = tree.findall('PROC_EVENT/WavFileHandler')
        for wfh in WavFileHandler:
            self.sampling_attr.update(wfh.attrib)

        # Info about the sampling
        self.fs = float(tree.find('CFG/FS').text)

        # Setup information. Read SoundTrap gain ('HIGH' or 'LOW')
        if self.type_start == 'NEW':
            self.st_gain = tree.find('EVENT/AUDIO').get('Gain')
        else:
            if last_gain is None:
                print('Unknown gain if it is reopened and the last gain is not passed!')
            self.st_gain = last_gain
        
        self.start_time = dt.datetime.strptime(self.sampling_attr['SamplingStartTimeLocal'], '%d/%m/%Y %H:%M:%S')
        self.stop_time = dt.datetime.strptime(self.sampling_attr['SamplingStopTimeLocal'], '%d/%m/%Y %H:%M:%S')


    def get_name_info(self, wavfile_name):
        """
        Get the data and time of recording from the name of the file 
        """
        name = wavfile_name.split('.')
        date_string = name[1]
        date = dt.datetime.strptime(date_string, "%y%m%d%H%M%S")

        return date


    def detect_possible_click_zones(self, lowcut, highcut, min_separation_time):
        """ 
        Detect a part of the signal with high energy at the desired band
        """
        if self.sound_file.frames <= self.max_num_samples:
            signal = self.sound_file.read()
            time_line, freq, cut_signal_total = self._detect_possible_click(signal, lowcut, highcut)
            
        else:       
            time = 0
            time_line = []
            cut_signal_total = []
            for block in self.sound_file.blocks(blocksize=self.max_num_samples):
                t, freq, cut_signal = self._detect_possible_click(block, lowcut, highcut)

                # Add the times and the signal
                converted_t = t + time
                time = converted_t[-1]
                time_line = np.concatenate((time_line, converted_t), axis=0)
                cut_signal_total = np.concatenate((cut_signal_total, cut_signal), axis=0)

        threshold = 1.1 * np.sqrt(cut_signal_total**2).mean()
        peaks_detection = time_line[np.where(cut_signal_total > threshold)]

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(time_line, cut_signal_total)
        ax[1].scatter(x=peaks_detection, y=np.ones(peaks_detection.shape))
        ax[0].set_xlim([time_line.min(), time_line.max()])
        ax[1].set_xlim([time_line.min(), time_line.max()])
        plt.show()
        plt.close()

        peaks_diff = np.diff(peaks_detection)
        time_parts = np.split(peaks_detection, np.where(peaks_diff > min_separation_time)[0] + 1)

        zones = []
        for part in time_parts: 
            if part.size > 1:
                zone = {'start': part[0], 'duration': part[-1] - part[0]}
                zones.append(zone)
        
        return zones


    def _detect_possible_click(self, signal, lowcut, highcut):
        """
        Detect possible clicks in the passed signal 
        """
        freq, t, spectra = self.spectrogram(signal)

        # Get only the part at the desired band
        cut_signal = spectra[np.where((freq > lowcut) & (freq < highcut))].mean(axis=0)

        return t, freq, cut_signal
    

    def zone2click(self, zone):
        """
        Return a click object of the specified zone
        """
        # Select a part multiple of 2 to the power 2 and read it (faster)
        duration = int(2 ** np.ceil(np.log2(zone['duration']*self.fs))) 
        start = max(int(zone['start']*self.fs - (duration - zone['duration']*self.fs)/2), 0)
        self.sound_file.seek(start)
        signal = self.sound_file.read(frames=duration)   

        timestamp = self.date + dt.timedelta(seconds=zone['start'])

        click = Click(signal, self.fs, timestamp)

        return click     



class Click:
    def __init__(self, sound_block, fs, timestamp, nfft=512, verbose=False):
        """
        Start a porpoise click
        """
        self.nfft = nfft
        self.fs = fs
        self.sound_block = sound_block
        self.timestamp = timestamp

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
    configfile_path = os.path.join('calibration', str(serial_number)+'.ini')
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
