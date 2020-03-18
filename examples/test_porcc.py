import pandas as pd
import numpy as np
import os
import soundfile as sf
import configparser
import pymongo
import json

import pyvalmongo

from pyporcc import mat2py
from pyporcc import porcc
from pyporcc import click_detector
from pyporcc import pamguard
from pyporcc import soundtrap



# Sound Files
soundtrap_folder_path = "C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/BelwindTest"
# soundtrap_folder_path = '//archive/cpod/Soundtrap/najaar2017_reefballs_Belwind'
# seiche_folder_path = "C:/Users/cleap/Documents/Data/Sound Data/Seiche/AutonautTest"

# PAMGuard SQLite3 output 
pamguard_db = 'C:/Users/cleap/Documents/Data/PAMGuard/click_detector.sqlite3'

# Hydrophone 
# name = 'Seiche'
# model = 'uPam'
# sensitivity = -196.0
# preamp_gain = 0.0
# Vpp = 2.0
# hydrophone = pyhy.Seiche(name, model, sensitivity, preamp_gain, Vpp)

name = 'SoundTrap'
model = 1
serial_number = 67416073
Vpp = 2
hydrophone = soundtrap_hf.SoundTrapHF(name, model, serial_number, Vpp)

# lowcutfreq = 100e3          # Lowcut frequency 
# highcutfreq = 160e3         # Highcut frequency
# min_separation_time = 0.1

# Cosentino data
models_config_path = 'pyporcc/models/log_models.ini'
train_hq_data_cosentino_path = 'pyporcc/data/cosentino/trainHQ_data.pkl'
train_lq_data_cosentino_path = 'pyporcc/data/cosentino/trainLQ_data.pkl'
test_data_cosentino_path = 'pyporcc/data/cosentino/test_data.pkl'
click_model_path = 'pyporcc/data/cosentino/standard_click.wav'
fs_data = 500000
nfft = 512

train_hq_data_path = 'pyporcc/data/clicks_hq.pkl'
train_lq_data_path = 'pyporcc/data/clicks_lq.pkl'
test_data_path = 'pyporcc/data/clicks_test.pkl'



# Some easy functions to get all the data set up
def calculate_clicks_params(fs_data, click_model_path, train_hq_data_path, train_lq_data_path, test_data_path):
    """
    Calculate the click parameters from the cosentino traning df and save them as a pickle
    The paths have to be pickle files with the necessary information. If they already have the parameters calculated, 
    they will be calculated using the python class and added with the "py" suffix to the df
    """
    df_hq, df_lq, df_test = mat2py.load_pickle_data([train_hq_data_path, train_lq_data_path, test_data_path])
    model = porcc.PorCCModel(train_hq_df=df_hq, train_lq_df=df_lq, test_df=df_test)
    
    clicks_hq_df = model.calculate_clicks_params(df_name='hq', fs=fs_data, click_model_path=click_model_path, save=True)
    clicks_lq_df = model.calculate_clicks_params(df_name='lq', fs=fs_data, click_model_path=click_model_path, save=True)
    clicks_test_df = model.calculate_clicks_params(df_name='test', fs=fs_data, click_model_path=click_model_path, save=True)

    return clicks_hq_df, clicks_lq_df, clicks_test_df


def create_and_save_models(fs_data, train_hq_data_path, train_lq_data_path, test_data_path):
    """
    Load the pickle data and calculate the models
    The df paths have to be with all the click parameters already calculated!
    """
    df_hq, df_lq, df_test = mat2py.load_pickle_data([train_hq_data_path, train_lq_data_path, test_data_path])
    models = porcc.PorCCModel(train_hq_df=df_hq, train_lq_df=df_lq, test_df=df_test)

    models.find_best_model('hq')
    models.find_best_model('lq')

    models.save('pyporcc/models/porcc_models.pkl', 'pickle')

    return models


def load_porcc_models(models_config_path, train_hq_data_path, train_lq_data_path, test_data_path):
    """
    Load the models coefficients
    """
    df_hq, df_lq, df_test = mat2py.load_pickle_data([train_hq_data_path, train_lq_data_path, test_data_path])
    models = porcc.PorCCModel(train_hq_df=df_hq, train_lq_df=df_lq, test_df=df_test)
    models.load_model_from_config(models_config_path)
    return models


def test_click_calculation(df_clicks, df_test, col_vars):
    """
    Test the calculation of the click parameters indicated in col_vars obtained with python compared to the ones obtained in the paper (on the Test DB)
    """
    # Compare each field
    rel_error = np.abs(df_test[col_vars] - df_clicks[col_vars])/df_test[col_vars].mean()
    mean_rel_error = rel_error.mean()
    
    return mean_rel_error


if __name__ == "__main__":
    """
    Start a PorCC study and apply the classifier to all the sound files 
    """
    ########### COMPUTE PARAMS
    # Load cosentino (already in pkl) files and calculate the params using pyporcc approach 
    # (make sure the calculation of params is not influencing)
    # calculate_clicks_params(fs_data, click_model_path, train_hq_data_cosentino_path, train_lq_data_cosentino_path, test_data_cosentino_path)

    ########### IF MODEL NOT TRAINED
    # Train the model and save it
    # models = create_and_save_models(fs_data, train_hq_data_path, train_lq_data_path, test_data_path)
    # models = load_porcc_models(models_config_path, train_hq_data_path, train_lq_data_path, test_data_path)

    # df_hq, df_lq, df_test = mat2py.load_pickle_data([train_hq_data_path, train_lq_data_path, test_data_path])
    # models = porpoise_classifier.PorpoiseClassifier()
    # train_df = models.join_train_data(df_hq, df_lq)
    # models.prepare_train_data(train_df)
    # models.prepare_test_data(df_test)

    # models_list = ['svc', 'logit', 'forest', 'knn']
    # models.train_models(models_list, standarize=True, feature_sel=True)

    # Test it
    # error, predicted = porcc_al.test_classification_vs_matlab(models.test_df)
    # print('The model gets a %s \% of good assignments' % (error*100))

    ########### IF MODEL TRAINED AND SAVED
    # models = pd.read_pickle('pyporcc/models/porcc_models.pkl')

    ########### PORCC from Cosentino
    # df_test = mat2py.load_pickle_data([test_data_path])[0]
    # porcc_al = porcc.PorCC(config_file=models_config_path, load_type='manual')
    # porcc_al.test_classification_vs_matlab(df_test)

    ########### CLASSIFY
    # Init classifier
    # porcc_al = porcc.PorCC(load_type='custom', hq_mod=models.hq_mod, lq_mod=models.lq_mod, hq_params=models.hq_params, lq_params=models.lq_params, fs=fs_data)
    
    # Run on sound data
    # cd = click_detector.ClickDetector()
    # cd.get_click_clips(hydrophone, folder_path)
    # clicks_df = cd.clicks_df(click_model_path)
    # pamguard_clicks = pamguard.read_clicks_output(pamguard_db)
    # classification = porcc_al.classify_matrix(clicks_df)

    # Read Sountrap output 
    clicks_df = hydrophone.read_HFfolder(soundtrap_folder_path, zip_mode=False)
    converter = click_detector.ClickConverter(click_model_path)
    clicks = converter.clicks_df(clicks_df, save=True, save_path='pyporcc/data/soundtrap_clicks.json')
    # clicks_df = pd.read_pickle('pyporcc/data/soundtrap_clicks.pkl')
    # classification = models.classify_matrix(clicks_df)



    print(clicks_df.size)