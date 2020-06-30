import os
import sys
import pandas as pd
from sklearn import model_selection

from pyporcc import porcc


# Cosentino Model
models_config_path = 'pyporcc/models/log_models.ini'

# All the data clicks
clicks_path = "C:/Users/cleap/Documents/Data/Sound Data/Clicks/soundtrap/soundtrap_clicks.pkl"

# Where to save the data to validate
validate_clicks_path = "C:/Users/cleap/Documents/Data/Sound Data/Clicks/soundtrap/validate_soundtrap_clicks.pkl"


if __name__ == "__main__":
    """
    Load the clicks and classify them 
    Once the classification is done, select only a portion to be validated
    """
    porcc_al = porcc.PorCC(load_type='manual', config_file=models_config_path)
    clicks_df = pd.read_pickle(clicks_path)

    clicks_class = porcc_al.classify_matrix(clicks_df)

    X = clicks_class[['wave', 'datetime', 'Q', 'duration', 'ratio', 'XC', 'CF', 'BW', 'const']]
    y = clicks_class['pyPorCC']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.5)

    X_train.to_pickle(validate_clicks_path)