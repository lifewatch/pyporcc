import pandas as pd

import pyhydrophone as pyhy

from pyporcc import porcc
from pyporcc import porpoise_classifier
from pyporcc import click_detector


# Cosentino data
click_model_path = 'pyporcc/data/cosentino/standard_click.wav'
fs_data = 500000

# Data with the params already calculated
train_hq_data_path = 'pyporcc/data/clicks_hq.pkl'
train_lq_data_path = 'pyporcc/data/clicks_lq.pkl'
test_data_path = 'pyporcc/data/clicks_test.pkl'

# Load data to train and thest the models
df_hq = pd.read_pickle(train_hq_data_path)
df_lq = pd.read_pickle(train_lq_data_path)
df_test = pd.read_pickle(test_data_path)


# In case the parameters are not calculated
def calculate_click_params(click_model_path, df_list):
    """
    Add the click parameters calculations to the df in case they have not been calculated yet 
    """
    converter = click_detector.ClickConverter(click_model_path)
    new_df = []
    for df in df_list: 
        new_df.append(converter.clicks_df(df))
    
    return new_df


def porcc_models(df_hq, df_lq, df_test):
    """
    The df have to be with all the click parameters already calculated!
    """
    models = porcc.PorCCModel(train_hq_df=df_hq, train_lq_df=df_lq, test_df=df_test)

    models.find_best_model('hq')
    models.find_best_model('lq')

    models.save('pyporcc/models/porcc_models.pkl', 'pickle')

    return models


def other_models(df_hq, df_lq, df_test, models_list):
    """
    Create the classification models
    The df have to be with all the click parameters already calculated!
    """
    models = porpoise_classifier.PorpoiseClassifier()
    train_df = models.join_train_data(df_hq, df_lq)
    models.prepare_train_data(train_df)
    models.prepare_test_data(df_test)

    models.train_models(models_list, standarize=True, feature_sel=True)

    return models



if __name__ == "__main__":
    """
    Start a PorCC study and apply the classifier to all the sound files 
    """
    # Train the model and save it
    models_porcc = porcc_models(train_hq_data_path, train_lq_data_path, test_data_path)

    models_list = ['svc', 'logit', 'forest', 'knn']
    models_custom = other_models(df_hq, df_lq, df_test, models_list)

    # Do whatever to test the models