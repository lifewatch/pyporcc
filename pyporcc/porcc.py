import numpy as np
import matplotlib.pyplot as plt 
import os
import configparser
import datetime as dt
import itertools
import pandas as pd
from scipy import signal as sig
import statsmodels.api as sm
from sklearn import metrics, linear_model
import pickle
import joblib

import pyporcc.click_detector as click_detector



class PorCCModel:
    def __init__(self, train_hq_df, train_lq_df, test_df):
        """
        Find the click model 
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
        Pick the right df
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


    def load_model_from_config(self, configfile_path):
        """
        Load PorCC model coefficients 
        """
        config = configparser.ConfigParser()
        config.read(configfile_path)

        logitCoefHQ = np.array(config['MODEL']['logitCoefHQ'].split(',')).astype(np.float)
        logitCoefLQ = np.array(config['MODEL']['logitCoefLQ'].split(',')).astype(np.float)
        
        self.hq_params = np.array(config['MODEL']['hq_params'].split(','))
        self.lq_params = np.array(config['MODEL']['lq_params'].split(','))

        # Starts and fit to get the classes 
        logit_hq = linear_model.LogisticRegression()
        reg_hq = logit_hq.fit(self.train_hq_df[self.hq_params], self.train_hq_df['P'])

        logit_lq = linear_model.LogisticRegression()
        reg_lq = logit_lq.fit(self.train_lq_df[self.lq_params], self.train_lq_df['P'])

        # Cheat and force the coefficients
        reg_hq.coef_ = np.array([logitCoefHQ[:-1]])
        reg_hq.intercept_ = logitCoefHQ[-1]
        self.hq_mod = reg_hq

        reg_lq.coef_ = np.array([logitCoefLQ[:-1]])
        reg_lq.intercept_=logitCoefLQ[-1]
        self.lq_mod = reg_lq


    def find_best_model(self, name):
        """
        Find the best model among the possible models
        'name' is HQ or LQ
        """
        # Get all the possible models
        models = self.find_possible_models(name)
        df = self.pick_df(name)

        # Select the appropiate columns combination according to AIC
        models = models[models[:,-1].argsort()]
        comb, mod, aic = models[0]

        y = df['P']
        X = df[comb]

        if name == 'hq':
            self.hq_mod = mod
            self.hq_params = X.columns
        else: 
            self.lq_mod = mod
            self.lq_params = X.columns

        print('The winning combination for %s is %s. AIC: %s' % (name, comb, aic))
        print(mod.summary())
        return X.columns, mod

                
    def find_possible_models(self, name):
        """
        Create all the regression models
        classification ('P') = 0:N, 1:LQ, 2:HQ
        'name' is HQ or LQ
        """
        models = []
        df = self.pick_df(name)

        # Go through all the possible combinations (from 1 to all the variables)
        for i in np.arange(1, len(self.ind_vars)+1):
            var_combinations = itertools.combinations(self.ind_vars, i)
            for comb in var_combinations:
                # Regression model
                y = df['P']
                X = df[list(comb)]
                
                logit = linear_model.LogisticRegression(max_iter=500, tol=1e-5, C=0.1)
                reg = logit.fit(X, y)

                # Calculate AIC
                prob = reg.predict_log_proba(X)
                llk = np.sum(y*prob[:,1] + (1-y)*prob[:,0])
                aic = 2*(len(comb)) - 2*(llk)

                # Append the model
                models.append([list(comb), reg, aic])

        return np.array(models)


    def test_model(self, logreg, X, y):
        """
        Test the model with the test data 
        """
        # Compute accuracy
        y_pred = logreg.predict(X)
        accuracy = logreg.score(X, y)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy))

        # Compute confusion matrix
        confusion_matrix = metrics.confusion_matrix(y, y_pred)
        print(confusion_matrix)

        # Plot ROC
        logit_roc_auc = metrics.roc_auc_score(y, logreg.predict(X))
        fpr, tpr, thresholds = metrics.roc_curve(y, logreg.predict_proba(X)[:,1])
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()

        return confusion_matrix

    
    def save(self, path, extension):
        """
        Save the current models in a file. Can be chosen to save it as pickle or joblib
        """
        if extension == 'pickle':
            pickle.dump(self, open(path, 'wb'))
        elif extension == 'joblib': 
            joblib.dump(self, path)
        else:
            raise Exception('This extension is unknown!')


    def calculate_clicks_params(self, df_name, fs, click_model_path, save=False):
        """
        Add to the df the click parameters calculated by the Click Class
        """
        # Pick the right df
        df = self.pick_df(df_name)

        # Pass the samplig frequency as metadata
        df.fs = fs
        
        # Init a converter to calculate all the params
        converter = click_detector.ClickConverter(click_model_path, self.ind_vars)
        df_clicks = converter.clicks_df(df)
        
        df_joint = df.join(df_clicks, lsuffix='_mat', rsuffix='')
        if save:
            df_joint.to_pickle('pyporcc/data/clicks_%s.pkl' % (df_name))

        return df_joint



class PorCC:
    def __init__(self, load_type, **kwargs):
        """
        Start the classifier 
        if load_type is set to 'manual', loads the models from the config file
        Then config_file must be specified

        if load_type is set to 'trained_model'
        Then hq_mod, lq_mod, hq_params, lq_params have to be specified
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
        Load the coefficients of the models to calculate the probablity manually 
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
        """
        x = pd.DataFrame(data={'Q':click.Q, 'duration':click.duration, 'ratio':click.ratio, 'XC':click.xc, 'CF':click.cf, 'PF':click.pf, 'BW':click.bw})
        porps = self.classify(x)

        return porps

    
    def classify_row(self, X):
        """
        Classify one row according to the params [PF, CF, Q, XC, duration, ratio, BW]
        """
        # Add the independent variable
        X['const'] = 1
        if (X['CF'] > self.lowcutfreq) and (X['CF'] < self.highcutfreq) and (X['Q'] > 4):
            # Evaluate the model on the given x
            prob_hq = self.hq_mod.predict_proba(X[self.hq_params].values.reshape(1,-1))[0][1]

            # Assign clip to a category
            if prob_hq >= self.th1:
                # HQ click
                porps = 1  
            else:
                prob_lq = self.lq_mod.predict_proba(X[self.lq_params].values.reshape(1,-1))[0][1]
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
        """
        # Add the independent variable for the regression
        df['const'] = 1

        # Initialize the prediction column
        df['pyPorCC'] = 0

        # Evaluate the model on the given x
        df['prob_hq'] = self.hq_mod.predict_proba(df[self.hq_params])[0][:,1]
        df['prob_lq'] = self.lq_mod.predict_proba(df[self.lq_params])[0][:,1]

        # Decide
        loc_idx = (df['CF'] > self.lowcutfreq) & (df['CF'] < self.highcutfreq) & (df['Q'] > 4)
        df.loc[~loc_idx, 'pyPorCC'] = 3                                                                      # N Clicks
        df.loc[loc_idx & (df['prob_hq'] > self.th1), 'pyPorCC'] = 1                                          # HQ Clicks
        df.loc[loc_idx & (df['prob_hq'] < self.th1) & (df['prob_lq'] > self.th2), 'pyPorCC'] = 2             # LQ Clicks
        df.loc[loc_idx & (df['prob_hq'] < self.th1) & (df['prob_lq'] <= self.th2), 'pyPorCC'] = 3            # N Clicks

        return df

    
    def test_classification(self, test_df, col_name='ClassifiedAs'):
        """
        Test the algorithm. With the same parameters, test the prediction output of the algorithm
        """
        predicted_df = self.classify_matrix(test_df)

        # Compare 'pyPorCC' vs 'ClassifiedAs' (MATLAB)
        error = np.sum(test_df[col_name] != predicted_df['pyPorCC'])/len(test_df)

        return error, predicted_df



class ManualLogit:
    def __init__(self, coef, th=0.5):
        """
        Init a logit probabilty prediction class
        """
        self.coef_ = coef
        self.th_ = th


    def predict(self, X):
        """
        Predict the classification of X
        """
        proba = self.predict_proba(X)[0][:,1]
        y_pred = np.zeros(proba.shape)
        y_pred[np.where(proba >= self.th_)] = 1

        return y_pred

    
    def predict_proba(self, X):
        """
        Predict the probability
        """
        z = np.sum(X*self.coef_[1:], axis=1) + self.coef_[0]
        prob_1 = 1/(1 + np.power(np.e, z))
        prob = np.column_stack((prob_1, 1-prob_1))

        return [prob]

    
    def predict_log_proba(self, X):
        """
        Predict the log probability
        """
        log_prob = np.log(self.predict_proba(X)[0])

        return log_prob
