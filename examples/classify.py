import os
import sys
import pandas as pd

from pyporcc import porcc


# Cosentino Model
models_config_path = 'pyporcc/models/log_models.ini'

# Already trained models 
models_trained_path = 'pyporcc/models/porcc_models.pkl'

# Data to classify
test_data_path = 'C:/Users/cleap/Documents/Data/Sound Data/Clicks/cosentino/porcc_params/clicks_test.pkl'


def load_porcc_classifier_coef(config_file):
    """
    Load the PorCC classifier from the coef configuration file 
    """
    porcc_al = porcc.PorCC(load_type='manual', config_file=models_config_path)
    return porcc_al


def load_porcc_classifier_trained(models_trained_path):
    """
    Load the PorCC classifier from the trained models saved as pickles
    Model saved in the model file must be of the PorCCModel class
    """
    models = pd.read_pickle(models_trained_path)
    porcc_al = porcc.PorCC(load_type='trained_model', hq_mod=models.hq_mod, lq_mod=models.lq_mod, hq_params=models.hq_params, lq_params=models.lq_params)
    return porcc_al



if __name__ == "__main__":
    """
    Start a PorCC study and apply the classifier to all the sound files 
    """
    # Load the classifier (choose one)
    porcc_al_coef = load_porcc_classifier_coef(config_file=models_config_path)
    porcc_al_trained = load_porcc_classifier_trained(models_trained_path)

    # Load data 
    clicks_df = pd.read_pickle(test_data_path)
    classification_trained = porcc_al_trained.classify_matrix(clicks_df)
    classification_coef = porcc_al_coef.classify_matrix(clicks_df)