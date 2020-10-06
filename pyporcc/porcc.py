#!/usr/bin/python
"""
Module : porcc.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Instituut voor de Zee)
Last Accessed : 9/23/2020
"""

from pyporcc import click_detector
from pyporcc import utils

import pickle
import joblib
import itertools
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt 

from sklearn import metrics, linear_model, model_selection, preprocessing

pd.plotting.register_matplotlib_converters()
plt.style.use('ggplot')


class PorCCModel:
    def __init__(self, train_hq_df, train_lq_df, test_df):
        """
        Find the click model
        Parameters
        ----------
        train_hq_df : DataFrame
            DataFrame with the rows used for training the HQ model
        train_lq_df : DataFrame
            DataFrame with the rows used for training the LQ model
        test_df : DataFrame
            DataFrame to test the models
        """
        self.ind_vars = ['Q', 'duration', 'ratio', 'XC', 'CF', 'BW']
        self.dep_vars = ['P']
        self.train_hq_df = train_hq_df
        self.train_lq_df = train_lq_df
        self.test_df = test_df

        # Model definition, initialized to None until calculated 
        self.hq_params = None
        self.hq_mod = None
        self.lq_params = None
        self.lq_mod = None

    def pick_df(self, name):
        """
        Pick the right DataFrame from the model
        Parameters
        ----------
        name : string
            String of the right dataframe to pick. It can be 'hq', 'lq' or 'test'

        Returns
        -------
        DataFrame correspondent to the name
        """
        if name == 'hq':
            df = self.train_hq_df
        elif name == 'lq':
            df = self.train_lq_df
        elif name == 'test':
            df = self.test_df 
        else:
            raise Exception('This is not a valid Data Frame name')

        return df
    
    def hq_lq_separation(self, df, train_size=0.2):
        """
        Separate the df in hq_train, lq_train and test
        Parameters
        ----------
        df : DataFrame
            Whole dataframe to be separated
        train_size : float
            From 0 to 1, percentage to use to train

        Returns
        -------
        DataFrame for hq model, DataFrame for loq model and DataFrame for testing
        """
        train, self.test_df = model_selection.train_test_split(df, train_size=train_size)
        self.train_hq_df = train[train['P'] != 2]
        self.train_lq_df = train[train['P'] != 1]
        self.train_hq_df['P'].replace(3, 0)
        self.train_lq_df['P'].replace(3, 0)
        self.train_lq_df['P'].replace(2, 1)

        return self.train_hq_df, self.train_lq_df, self.test_df

    def load_model_from_config(self, configfile_path):
        """
        Load PorCC model coefficients
        Parameters
        ----------
        configfile_path : string or Path
            .ini file where the coefficients of the LQ and the HQ models are specified
        """
        config = configparser.ConfigParser()
        config.read(configfile_path)

        logitcoef_hq = np.array(config['MODEL']['logitCoefHQ'].split(',')).astype(np.float)
        logitcoef_lq = np.array(config['MODEL']['logitCoefLQ'].split(',')).astype(np.float)
        
        self.hq_params = np.array(config['MODEL']['hq_params'].split(','))
        self.lq_params = np.array(config['MODEL']['lq_params'].split(','))

        # Starts and fit to get the classes 
        logit_hq = linear_model.LogisticRegression()
        reg_hq = logit_hq.fit(self.train_hq_df[self.hq_params], self.train_hq_df['P'])

        logit_lq = linear_model.LogisticRegression()
        reg_lq = logit_lq.fit(self.train_lq_df[self.lq_params], self.train_lq_df['P'])

        # Cheat and force the coefficients
        reg_hq.coef_ = np.array([logitcoef_hq[:-1]])
        reg_hq.intercept_ = logitcoef_hq[-1]
        self.hq_mod = reg_hq

        reg_lq.coef_ = np.array([logitcoef_lq[:-1]])
        reg_lq.intercept_ = logitcoef_lq[-1]
        self.lq_mod = reg_lq

    def find_best_model(self, name):
        """
        Find the best model among the possible models
        Parameters
        ----------
        name : string
            Set to 'HQ' or 'LQ' to find the according model

        Returns
        -------
        columns, model. Columns are the names of the relevant parameters of the model. Model is an object instance
        """
        # Get all the possible models
        models = self.find_possible_models(name)
        df = self.pick_df(name)

        # Select the appropriate columns combination according to AIC
        models = models[models[:, -1].argsort()]
        comb, mod, aic = models[0]

        # y = df['P']
        x = df[comb]

        if name == 'hq':
            self.hq_mod = mod
            self.hq_params = x.columns
        else: 
            self.lq_mod = mod
            self.lq_params = x.columns

        print('The winning combination for %s is %s. AIC: %s' % (name, comb, aic))
        # print(mod.summary())
        return x.columns, mod
                
    def find_possible_models(self, name):
        """
        Create all the regression models
        classification ('P') = 0:N, 1:LQ, 2:HQ

        Parameters
        ----------
        name : string
            Set to 'HQ' or 'LQ' to find the according model

        Returns
        -------
        List of all the possible models as instances (alreay fitted)
        """
        models = []
        df = self.pick_df(name)

        # Go through all the possible combinations (from 1 to all the variables)
        for i in np.arange(1, len(self.ind_vars)+1):
            var_combinations = itertools.combinations(self.ind_vars, i)
            for comb in var_combinations:
                # Regression model
                y = df['P']
                x = df[list(comb)]
                
                logit = linear_model.LogisticRegression(max_iter=500, tol=1e-5, C=0.1)
                reg = logit.fit(x, y)

                # Calculate AIC
                y_prob = reg.predict_proba(x)
                aic = utils.aic_score(y, y_prob, len(comb))

                # Append the model
                models.append([list(comb), reg, aic])

        return np.array(models)
    
    def save(self, save_path):
        """
        Save the current models in a file. Can be chosen to save it as pickle or joblib

        Parameters
        ----------
        save_path : string
            Path where to save the models
        """
        extension = save_path.split('.')[-1]
        if extension == 'pkl':
            pickle.dump(self, open(save_path, 'wb'))
        elif extension == 'joblib': 
            joblib.dump(self, save_path)
        else:
            raise Exception('This extension is unknown!')

    def calculate_clicks_params(self, df_name, fs, click_model_path, save_path=None):
        """
        Add to the df the click parameters calculated by the Click Class

        Parameters
        ----------
        df_name : string
            name of the dataframe to pick. It can be 'hq', 'lq' or 'test'
        fs : int
            Sampling frequency of the dataframe entries
        click_model_path : string or Path
            Path to the wav file containing the click model
        save_path : string or Path
            Path where to save the output (should be pickle extension)
        """
        # Pick the right df
        df = self.pick_df(df_name)

        # Pass the sampling frequency as metadata
        df.fs = fs
        
        # Init a converter to calculate all the params
        converter = click_detector.ClickConverter(click_model_path, self.ind_vars)
        df_clicks = converter.clicks_df(df)
        
        df_joint = df.join(df_clicks, lsuffix='_mat', rsuffix='')
        if save_path is not None:
            df_joint.to_pickle(save_path)

        return df_joint


class PorCC:
    def __init__(self, load_type, **kwargs):
        """
        Start the classifier
        If load_type is set to 'manual', loads the models from the config file. Then config_file must be specified
        If load_type is set to 'trained_model', loads the trained model.
        Then hq_mod, lq_mod, hq_params, lq_params have to be specified

        load_type : string
            'manual' or 'trained_model'
        """
        self.th1 = 0.9999                # threshold for HQ clicks
        self.th2 = 0.55                  # threshold for LQ clicks
        self.lowcutfreq = 100e3          # Lowcut frequency
        self.highcutfreq = 160e3         # Highcut frequency

        self.load_type = load_type
        if load_type == 'manual':
            self.manual_models(kwargs['config_file'])
        else:
            for key, val in kwargs.items(): 
                self.__dict__[key] = val

    def manual_models(self, configfile_path):
        """
        Load the coefficients of the models to calculate the probability manually

        Parameters
        ----------
        configfile_path : string or Path
            Path to the config file with the coefficients for the models
        """
        config = configparser.ConfigParser()
        config.read(configfile_path)

        hq_coef = np.array(config['MODEL']['logitCoefHQ'].split(',')).astype(np.float)
        lq_coef = np.array(config['MODEL']['logitCoefLQ'].split(',')).astype(np.float)

        self.hq_mod = ManualLogit(hq_coef)
        self.lq_mod = ManualLogit(lq_coef)
        
        self.hq_params = np.array(config['MODEL']['hq_params'].split(','))
        self.lq_params = np.array(config['MODEL']['lq_params'].split(','))

    def classify_click(self, click):
        """
        Classify the click in HQ, LQ, N

        Parameters
        __________
        click : Click object
            Click to classify
        """
        x = pd.DataFrame(data={'Q': click.Q, 'duration': click.duration, 'ratio': click.ratio, 'XC': click.xc,
                               'CF': click.cf, 'PF': click.pf, 'BW': click.bw})
        porps = self.classify_row(x)

        return porps
    
    def classify_row(self, x):
        """
        Classify one row according to the params [PF, CF, Q, XC, duration, ratio, BW]

        Parameters
        ----------
        x : pandas row or dictionary
            Row to be classified
        """
        # Add the independent variable
        x['const'] = 1
        if (x['CF'] > self.lowcutfreq) and (x['CF'] < self.highcutfreq) and (x['Q'] > 4):
            # Evaluate the model on the given x
            prob_hq = self.hq_mod.predict_proba(x[self.hq_params].values.reshape(1, -1))[0][1]

            # Assign clip to a category
            if prob_hq >= self.th1:
                # HQ click
                porps = 1  
            else:
                prob_lq = self.lq_mod.predict_proba(x[self.lq_params].values.reshape(1, -1))[0][1]
                if prob_lq > self.th2:
                    # LQ click
                    porps = 2  
                else:
                    # HF Noise
                    porps = 3  
        else:
            porps = 3

        return porps        

    def classify_matrix(self, df):
        """
        Classify according to the params [PF, CF, Q, XC, duration, ratio, BW]

        Parameters
        ----------
        df : DataFrame
            DataFrame to be classified. Parameters PF, CF, Q, XC, duration, ratio and BW must be specified
        """
        # Add the independent variable for the regression
        df = df.assign(const=1)

        # Initialize the prediction column
        df = df.assign(pyPorCC=0)

        # Evaluate the model on the given x
        df = df.assign(prob_hq=self.hq_mod.predict_proba(df[self.hq_params])[:, 1])
        df = df.assign(prob_lq=self.lq_mod.predict_proba(df[self.lq_params])[:, 1])

        # Decide
        loc_idx = (df['CF'] > self.lowcutfreq) & (df['CF'] < self.highcutfreq) & (df['Q'] > 4)
        df.loc[~loc_idx, 'pyPorCC'] = 3                                                                      # N Clicks
        df.loc[loc_idx & (df['prob_hq'] > self.th1), 'pyPorCC'] = 1                                          # HQ Clicks
        df.loc[loc_idx & (df['prob_hq'] < self.th1) & (df['prob_lq'] > self.th2), 'pyPorCC'] = 2             # LQ Clicks
        df.loc[loc_idx & (df['prob_hq'] < self.th1) & (df['prob_lq'] <= self.th2), 'pyPorCC'] = 3            # N Clicks

        return df
    
    def predict(self, df):
        """
        Classify and return the y value

        Parameters
        ----------
        df : DataFrame

        """
        y_pred = self.classify_matrix(df)['pyPorCC']

        return y_pred    

    def predict_proba(self, df):
        """
        Return the probability of being classified as HQ or LQ

        Parameters
        ----------
        df : DataFrame
            DataFrame to predict. Needs to have the columns passed as hq and lq parameters

        Returns
        -------
        DataFrame with prob_hq, prob_lq and total PorCC probability
        """
        # Add the independent variable for the regression
        df = df.assign(const=1)

        # Initialize the prediction column
        df = df.assign(prob=0)

        # Evaluate the model on the given x
        df = df.assign(prob_hq=self.hq_mod.predict_proba(df[self.hq_params])[:, 1])
        df = df.assign(prob_lq=self.lq_mod.predict_proba(df[self.lq_params])[:, 1])

        # Decide
        loc_idx = (df['CF'] > self.lowcutfreq) & (df['CF'] < self.highcutfreq) & (df['Q'] > 4)
        df.loc[~loc_idx, 'prob'] = 0                                                                         
        df.loc[loc_idx & (df['prob_hq'] > self.th1), 'prob'] = df.loc[loc_idx & (df['prob_hq'] > self.th1), 'prob_hq']   
        df.loc[loc_idx & (df['prob_hq'] < self.th1), 'prob'] = df.loc[loc_idx & (df['prob_hq'] < self.th1), 'prob_lq']                           

        return df[['prob_hq', 'prob_lq', 'prob']]          

    def test_classification(self, test_df, col_name='ClassifiedAs'):
        """
        Test the algorithm. With the same parameters, test the prediction output of the algorithm

        Parameters
        ----------
        test_df : DataFrame
            DataFrame to test, with the necessary columns for classify
        col_name : string
            Column name of the correct classification
        """
        predicted_df = self.classify_matrix(test_df)

        # Compare 'pyPorCC' vs 'ClassifiedAs' (MATLAB)
        error = np.sum(test_df[col_name] != predicted_df['pyPorCC'])/len(test_df)

        return error, predicted_df

    def plot_roc_curves(self, df, dep_var='ManualAsign'):
        """
        Plot the ROC curves for HQ, LQ and All

        Parameters
        ---------
        df : DataFrame
            DataFrame with the classification
        dep_var : string
            Name of the column where the classification
        """
        fig, ax = plt.subplots(1, 3)
        # Filter the data frame so it only has HQ-Noise, LQ-Noise and All-Noise
        hq_noise = df.loc[df[dep_var] != 2]
        lq_noise = df.loc[df[dep_var] != 1]
        self._plot_roc_curves(df=hq_noise, dep_var=dep_var, ax0=ax[0])
        ax[0].set_title('HQ vs Noise')
        self._plot_roc_curves(df=lq_noise, dep_var=dep_var, ax0=ax[1])
        ax[1].set_title('LQ vs Noise')
        self._plot_roc_curves(df=df, dep_var=dep_var, ax0=ax[2])
        ax[2].set_title('All vs Noise')

        plt.tight_layout()
        plt.show()
        plt.close()

    def _plot_roc_curves(self, df, dep_var='ManualAsign', ax0=None):
        """`    
        Plot the ROC curves for each part of the algorithm

        Parameters
        ----------
        df : DataFrame
            DataFrame with the information
        dep_var : string
            Name of the column of the dependent variable where the correct labels are
        """
        y_test = self.convert2binary(df[dep_var])
        if ax0 is None: 
            fig, ax = plt.subplots()
        else:
            ax = ax0
        probs = self.predict_proba(df)

        # Plot ROC
        fpr_hq, tpr_hq, thresholds = metrics.roc_curve(y_test, probs['prob_hq'], drop_intermediate=True)
        fpr_lq, tpr_lq, thresholds = metrics.roc_curve(y_test, probs['prob_lq'], drop_intermediate=True)
        fpr_all, tpr_all, thresholds = metrics.roc_curve(y_test, probs['prob'], drop_intermediate=True)

        ax.plot(fpr_hq, tpr_hq, label='PorCC HQ')
        ax.plot(fpr_lq, tpr_lq, label='PorCC LQ')
        ax.plot(fpr_all, tpr_all, label='PorCC All')

        if ax0 is None: 
            plt.tight_layout()
            plt.show()
            plt.close()
        
        return ax

    @staticmethod
    def convert2binary(y):
        """
        Return the y sample as binary (no porpoise / porpoise)

        Parameters
        ----------
        y : np.array
            Array with 3 classes to be converted to 2 classes (from Noise, LQ and HQ to Porpoise/No Porpoise)
        """
        y = y.replace(3, 0)
        y = y.replace(2, 1)  

        return y


class ManualLogit:
    def __init__(self, coef, th=0.5):
        """
        Init a logit probability prediction class

        Parameters
        ----------
        coef : list or np.array
            Coefficients of the model
        th : float
            From 0 to 1 threshold to make the classification decision
        """
        self.coef_ = coef
        self.th_ = th

    def predict(self, x):
        """
        Predict the classification of X

        Parameters
        ----------
        x : np.array
            Array with the X coefficients to be predict the class
        """
        x = preprocessing.StandardScaler(x)
        proba = self.predict_proba(x)[0][:, 1]
        y_pred = np.zeros(proba.shape)
        y_pred[np.where(proba >= self.th_)] = 1
        return y_pred

    def predict_proba(self, x):
        """
        Predict the probability with the coefficients

        Parameters
        ----------
        x : np.array
            Array with the X coefficients to be predict the probability
        """
        lower_bnd = np.log(np.finfo(np.float64).eps)
        upper_bnd = -lower_bnd
        x_scale = preprocessing.scale(x)
        xb = np.array(np.sum(x_scale*self.coef_[1:], axis=1) + self.coef_[0])
        odds = np.exp(-utils.constrain(xb, lower_bnd, upper_bnd))
        prob_1 = 1 / (1 + odds)
        prob = np.column_stack((1-prob_1, prob_1))
        return prob

    def predict_log_proba(self, x):
        """
        Predict the log probability

        Parameters
        ----------
        x : np.array
            Array with the X coefficients to be predict the logarithmic probability
        """
        prob = self.predict_proba(x)
        log_prob = np.log(prob)

        return log_prob





