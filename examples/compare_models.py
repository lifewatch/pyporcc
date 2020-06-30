import os
import sys
import pickle
import pandas as pd

from pyporcc import porcc, porpoise_classifier


# Cosentino Model
models_config_path = 'pyporcc/models/log_models.ini'

# Data with the params already calculated
train_hq_data_path = 'C:/Users/cleap/Documents/Data/Sound Data/Clicks/cosentino/porcc_params/clicks_hq.pkl'
train_lq_data_path = 'C:/Users/cleap/Documents/Data/Sound Data/Clicks/cosentino/porcc_params/clicks_lq.pkl'
test_data_path = 'C:/Users/cleap/Documents/Data/Sound Data/Clicks/cosentino/TestingData.pkl'
join_data_path = 'C:/Users/cleap/Documents/Data/Sound Data/Clicks/cosentino/porcc_params/clicks_cosentino.pkl'


# Load data to train and thest the models
df_hq = pd.read_pickle(train_hq_data_path)
df_lq = pd.read_pickle(train_lq_data_path)
df_test = pd.read_pickle(test_data_path)
df_join = pd.read_pickle(join_data_path)

# List of the other models which can be tested
models_list = ['svc', 'logit', 'forest', 'knn']


def load_porcc_classifier_coef(config_file):
    """
    Load the PorCC classifier from the coef configuration file 
    """
    porcc_al = porcc.PorCC(load_type='manual', config_file=models_config_path)
    return porcc_al


def load_porcc_classifier_trained(file_path):
    """
    Load the porcc classifier trained
    """
    f = open(file_path, 'rb')
    porcc_model = pickle.load(f)
    porcc_al = porcc.PorCC(load_type='trained_model', hq_mod=porcc_model.hq_mod, lq_mod=porcc_model.lq_mod, \
                            hq_params=porcc_model.hq_params, lq_params=porcc_model.lq_params)

    return porcc_al


def porcc_models(df_hq, df_lq, df_test):
    """
    The df have to be with all the click parameters already calculated!
    """
    models = porcc.PorCCModel(train_hq_df=df_hq, train_lq_df=df_lq, test_df=df_test)

    models.find_best_model('hq')
    models.find_best_model('lq')

    models.save('pyporcc/models/porcc_models.pkl')

    return models


def other_models(clicks_df, models_list, dep_var):
    """
    Create the classification models
    The df have to be with all the click parameters already calculated!
    """
    models = porpoise_classifier.PorpoiseClassifier(dep_var=dep_var)
    train_df, test_df = models.split_data(clicks_df, train_size=0.2)
    models.train_models(models_list, binary=True, standarize=True, feature_sel=False)

    return train_df, test_df, models


def load_model_files(models_list, dep_var):
    """
    Load the models from the pkl files saved
    """
    models = porpoise_classifier.PorpoiseClassifier(dep_var=dep_var)
    for model_name in models_list: 
        f = open('pyporcc/models/%s.pkl' % (model_name), 'rb')
        model = pickle.load(f)
        models.add_model(model_name=model_name, model=model, ind_vars=models.ind_vars, binary=True)
    
    return models



if __name__ == "__main__":
    """
    Start a PorCC study and apply the classifier to all the sound files 
    """
    # Load data 
    clicks_df = pd.read_pickle(test_data_path)
    clicks_df = clicks_df[clicks_df.ManualAsign != 0]

    # Load the PorCC classifier
    porcc_al_coef = load_porcc_classifier_coef(config_file=models_config_path)
    # porcc_al_model = load_porcc_classifier_trained('pyporcc/models/porcc_models.pkl')

    # Train the model and save it

    # train_df, test_df, models = other_models(clicks_df, models_list)
    # models.add_model(model_name='porCC', model=porcc_al_coef, ind_vars=porcc_al_coef.lq_params)
    
    # Load the models from the pickle files 
    models = load_model_files(models_list, dep_var='ManualAsign')
    models.test_data = clicks_df
    # Test all the models
    # results = models.test_models()
    models.plot_roc_curves(porcc_al_coef)