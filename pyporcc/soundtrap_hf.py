import numpy as np
import matplotlib.pyplot as plt 
import os
import datetime as dt
from scipy import signal as sig
import soundfile as sf
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile

import pyhydrophone as pyhy


class SoundTrapHF(pyhy.SoundTrap): 
    def __init__(self, name, model, serial_number, Vpp):
        """
        Init a SoundTrap HF reader
        """
        super().__init__(name, model, serial_number, Vpp)
    

    def read_HFfolder(self, main_folder_path, zip_mode=False):
        """
        Read all the clicks in all the folders
        """
        clicks = pd.DataFrame()
        for folder_name in os.listdir(main_folder_path):
            folder_path = os.path.join(main_folder_path, folder_name)
            folder_clicks = self.read_HFclicks(folder_path, zip_mode=zip_mode)
            clicks = clicks.append(folder_clicks, ignore_index=True)
        
        # Keep the metadata
        clicks.fs = folder_clicks.fs 

        return clicks


    def read_HFclicks(self, folder_path, zip_mode=False):
        """
        Read all the clicks stored in a folder with soundtrap files
        """
        clicks = pd.DataFrame()
        if zip_mode:
            folder_zip = zipfile.ZipFile(folder_path, 'r', allowZip64=True)
            files_list = folder_zip.namelist()
        else:
            files_list = os.listdir(folder_path)
        
        for file_name in files_list:
            extension = file_name.split(".")[-1]
            if extension == 'wav':
                bcl_name = file_name.replace('.wav', '.bcl')
                dwv_name = file_name.replace('.wav', '.dwv')
                xml_name = file_name.replace('.wav', '.log.xml')
                if zip_mode: 
                    bcl_path = folder_zip.open(bcl_name)
                    dwv_path = folder_zip.open(dwv_name)
                    xml_path = folder_zip.open(xml_name)
                else:
                    bcl_path = os.path.join(folder_path, bcl_name)
                    dwv_path = os.path.join(folder_path, dwv_name)
                    xml_path = os.path.join(folder_path, xml_name)

                try:
                    file_clicks = self._read_HFclicks(bcl_path, dwv_path, xml_path)
                    clicks = clicks.append(file_clicks, ignore_index=True)
                    fs = file_clicks.fs        
                except:
                    print(dwv_path, 'has some problem and can not be read')

        # Keep the metadata
        clicks.fs = fs

        return clicks


    def _read_HFclicks(self, bcl_path, dwv_path, xml_path):
        """
        Read the clicks of one soundtrap file 
        """

        # Read the wav file with all the clicks 
        sound_file = sf.SoundFile(dwv_path, 'r')

        # click_len has to be checked automatically
        click_len = self.read_HFparams(xml_path)

        # Read the info of clicks
        clicks_info = pd.read_csv(bcl_path)
        clicks_info = clicks_info[clicks_info['report'] == 'D']
        clicks_info = clicks_info[clicks_info['state'] == 1]
        waves = []
        
        for block in sound_file.blocks(blocksize=click_len):
            waves.append(block.astype(np.float))
        
        print(dwv_path, len(clicks_info), len(waves))

        if len(waves) < len(clicks_info):
            # Cut the clicks info if there are not enough snippets
            clicks_info = clicks_info.loc[0:len(waves)]
        
        clicks_info['wave'] = waves[0:len(clicks_info)]

        clicks_info['datetime'] = pd.to_datetime(clicks_info['rtime'] + clicks_info['mticks']/1e6)

        # Append the samplerate as metadata to be able to access it later
        clicks_info.fs = sound_file.samplerate

        return clicks_info

    
    def read_HFparams(self, xml_path):
        """
        Return the length of the clips and the time in between
        """
        tree = ET.parse(xml_path)

        # blank_len = int(tree.find('CFG/BLANKING').text)
        clip_len = int(tree.find('CFG/PREDET').text) + int(tree.find('CFG/POSTDET').text)

        return clip_len
