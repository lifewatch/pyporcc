import pandas as pd
import numpy as np
import os
import zipfile
import pyhydrophone as pyhy

import mat2py
import porcc
import click_detector


# Sound Files
# folder_path = "C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/BelwindTest"
folder_path = "C:/Users/cleap/Documents/Data/Sound Data/Seiche/AutonautTest"

# Hydrophone 
name = 'Seiche'
model = 'uPam'
sensitivity = -196.0
preamp_gain = 0.0
Vpp = 2.0
hydrophone = pyhy.Seiche(name, model, sensitivity, preamp_gain, Vpp)

# lowcutfreq = 100e3          # Lowcut frequency 
# highcutfreq = 160e3         # Highcut frequency
# min_separation_time = 0.1

# Cosentino data
models_path = 'pyporcc/log_models/models.ini'
hq_cosentino_model_path = 'pyporcc/data/cosentino/trainHQ_data.pkl'
lq_cosentino_model_path = 'pyporcc/data/cosentino/trainLQ_data.pkl'
test_cosentino_model_path = 'pyporcc/data/cosentino/test_data.pkl'
click_model_path = 'pyporcc/data/cosentino/standard_click.wav'
fs_data = 500000
nfft = 512

hq_model_path = 'pyporcc/data/clicks_hq.pkl'
lq_model_path = 'pyporcc/data/clicks_lq.pkl'
test_model_path = 'pyporcc/data/clicks_test.pkl'



# Some easy functions to get all the data set up
def calculate_click_params(fs_data, click_model_path, hq_model_path, lq_model_path, test_model_path):
    """
    Calculate the click parameters from the cosentino traning df and save them as a pickle
    The paths have to be pickle files with the necessary information. If they already have the parameters calculated, 
    they will be calculated using the python class and added with the "py" suffix to the df
    """
    df_hq, df_lq, df_test = mat2py.load_pickle_data([hq_model_path, lq_model_path, test_model_path])
    model = porcc.ClickModel(train_hq_df=df_hq, train_lq_df=df_lq, test_df=df_test)
    
    clicks_hq_df = model.calculate_clicks_params(df_name='hq', fs=fs_data, click_model_path=click_model_path, save=True)
    clicks_lq_df = model.calculate_clicks_params(df_name='lq', fs=fs_data, click_model_path=click_model_path, save=True)
    clicks_test_df = model.calculate_clicks_params(df_name='test', fs=fs_data, click_model_path=click_model_path, save=True)

    return clicks_hq_df, clicks_lq_df, clicks_test_df


def create_and_save_models(fs_data, hq_model_path, lq_model_path, test_model_path):
    """
    Load the pickle data and calculate the models
    The df paths have to be with all the click parameters already calculated!
    """
    df_hq, df_lq, df_test = mat2py.load_pickle_data([hq_model_path, lq_model_path, test_model_path])
    models = porcc.ClickModel(train_hq_df=df_hq, train_lq_df=df_lq, test_df=df_test)

    models.find_best_model('hq')
    models.find_best_model('lq')

    models.save('pyporcc/models/porcc_models.pkl', 'pickle')

    return models


def test_click_calculation(df_clicks, df_test, col_vars):
    """
    Test the calculation of the click parameters indicated in col_vars obtained with python compared to the ones obtained in the paper (on the Test DB)
    """
    # Compare each field
    rel_error = np.abs(df_test[col_vars] - df_clicks[col_vars])/df_test[col_vars].mean()
    mean_rel_error = rel_error.mean()
    
    return mean_rel_error


def get_click_detectors_df_soundtrap(folder_path, hydrophone, click_model_path, save=True):
    """
    Go through all the sound files and create a database with the detected clicks
    """
    clicks_df = pd.DataFrame(columns=['datetime', 'Q', 'duration', 'ratio', 'XC', 'CF', 'BW'])
    cd = click_detector.ClickDetector()

    # Run the classifier in every file of the folder
    for day_folder_name in os.listdir(folder_path):
        day_folder_path = os.path.join(folder_path, day_folder_name)
        zip_day_folder = zipfile.ZipFile(day_folder_path, 'r', allowZip64=True)

        for file_name in zip_day_folder.namelist():
            extension = file_name.split(".")[-1]
            if extension == 'wav':
                print(file_name)
                # Read both the wav and the xml file with the same name
                wav_file = zip_day_folder.open(file_name)
                xml_file = zip_day_folder.open(file_name.replace('wav', 'log') + '.xml')
                date = hydrophone.get_name_date(file_name)
                cd.detect_click_clips(file_path, date) 
    
    if save:
        clicks_df = cd.clicks_df(click_model_path)
        clicks_df.to_pickle('soundtrap_clicks.pkl')  
    
    return clicks_df  


def get_click_clips(folder_path, hydrophone, save=True):
    """
    Go through all the sound files and create a database with the detected clicks
    """
    # Start the click detector with the default values
    cd = click_detector.ClickDetector()          

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        extension = file_name.split(".")[-1]
        if extension == 'wav':
            print(file_name)
            date = hydrophone.get_name_date(file_name)
            cd.detect_click_clips(file_path, date)
             
    return cd



if __name__ == "__main__":
    """
    Start a PorCC study and apply the classifier to all the sound files 
    """
    # # Options to create files
    # calculate_click_params(fs_data, click_model_path, hq_cosentino_model_path, lq_cosentino_model_path, test_cosentino_model_path)
    models = pd.read_pickle('pyporcc/models/porcc_models.pkl')

    # # Options to test the model
    # models = create_and_save_models(fs_data, hq_model_path, lq_model_path, test_model_path)
    porcc_al = porcc.PorCC(load_type='custom', hq_mod=models.hq_mod, lq_mod=models.lq_mod, hq_params=models.hq_params, lq_params=models.lq_params, fs=fs_data)
    
    # # Run on the test data
    # error, predicted = porcc_al.test_classification_vs_matlab(models.test_df)
    # print('The model gets a %s \% of good assignments' % (error*100))
    
    # Run on sound data
    cd = get_click_clips(folder_path, hydrophone)
    clicks_df = cd.clicks_df(click_model_path)
    classification = porcc_al.classify_matrix(clicks_df)