import os
import sys
import pickle
import pandas as pd

from pyporcc import porcc, porpoise_classifier


# CPOD data
cpod_path = "C:/Users/cleap/Documents/Data/Sound Data/Clicks/CPOD/2017_Belwind_reefballs/CPOD_Belwind.csv"

# RAW SoundTrap Data (already classified)
soundtrap_porcc_path = "C:/Users/cleap/Documents/Data/Sound Data/Clicks/soundtrap/soundtrap_clicks_validate.pkl"

# PAMGuard SoundTrap output
pamguard_path = "C:/Users/cleap/Documents/Data/PAMGuard/PAMsqlite/belwind_soundtrap_clicks.csv"


period = ['09/08/2017', '01/09/2017']


def clicks_per_minute(df_list, period):
    """
    Plot the clicks per minute per df
    """


def _clicks_per_minute(df, period):
    """
    Get the clicks per minute of the df
    """



if __name__ == "__main__":
    """
    Read all the data and compare the detections
    """
    df_cpod = pd.read_csv(cpod_path)
    df_soundtrap_porcc = pd.read_pickle(soundtrap_porcc_path)
    df_pamguard = pd.read_csv(pamguard_path)

    df_list = [df_cpod, df_soundtrap_porcc, df_pamguard]

    clicks_per_minute(df_list)