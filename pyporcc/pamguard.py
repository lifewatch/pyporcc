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
import sqlite3

import pyhydrophone


class PAMGuard: 
    def __init__(self):
        """
        Init a PAMGuard class
        """
        self.name = 'PAMGUARD'


    def read_clicks_output(self, db_path):
        """
        Read the PAMGuard output from the sqlite db in the path
        """
        con = sqlite3.connect(db_path)
        clicks = pd.read_sql(sql='SELECT * from Click_Detector_Clicks', con=con)
        
        return clicks