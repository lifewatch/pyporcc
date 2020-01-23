import numpy as np
import matplotlib.pyplot as plt 
import os
import configparser
import datetime as dt
import scipy.io as sio
import itertools
import pandas as pd
from scipy import signal as sig
import statsmodels.api as sm
# from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
import joblib

import sound_click



class ClickModel:
    def __init__(self, click_model_path, fs, train_hq_df, train_lq_df, test_df):
        """
        Find the click model 
        """
        self.ind_vars = ['Q', 'duration', 'ratio', 'XC', 'CF', 'BW']
        self.dep_vars = ['P']
        fs_model, self.click_model = sio.wavfile.read(click_model_path)
        if fs_model != fs: 
            raise Exception('This click is not recorded at the same frequency than the classified data!')
        else:
            self.fs = fs
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
        comb = models[0][0]
        print('The winning combination for %s is %s' % (name, comb))
        y = df['P']
        X = df[comb]

        # Create the logit model
        logreg = LogisticRegression()
        logreg.fit(X, y)

        if name == 'hq':
            self.hq_mod = logreg
            self.hq_params = comb
        else: 
            self.lq_mod = logreg
            self.lq_params = comb

        return comb, logreg

                
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
                # print(comb)
                # Regression model
                y = df['P']
                X = df[list(comb)]
                logit_model = sm.genmod.GLM(endog=y, exog=X, family=sm.families.family.Binomial())
                reg = logit_model.fit()
                if not np.isnan(reg.aic):
                    models.append([list(comb), reg, reg.aic])

        return np.array(models)


    def test_model(self, X, y):
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


    def calculate_clicks_params(self, df_name, save=False):
        """
        Add to the df the click parameters calculated by the Click Class
        """
        df = self.pick_df(df_name)
        df_clicks = pd.DataFrame()

        # Calculate all the independent variables and append them to the df
        for var in self.ind_vars:
            df_clicks[var] = 0.00

        for idx in df.index:
            signal = df.loc[idx, 'wave'][:,0]
            click = sound_click.Click(signal, self.fs, df.loc[idx, 'datetime'], verbose=False)
            x_coeff = np.correlate(click.sound_block, self.click_model)
            xc = x_coeff.max()
            # BW =  powerbw(click, Fs)/1000
            df_clicks.loc[idx, self.ind_vars] = [click.Q, click.duration, click.ratio, xc, click.cf, click.bw]
        
        df_joint = df.join(df_clicks, lsuffix='_mat', rsuffix='')
        if save:
            df_joint.to_pickle('pyporcc/data/clicks_%s.pkl' % (df_name))

        return df_joint


class PorCC:
    def __init__(self, load_type, **kwargs):
        """
        Start the classifier 
        if load_type is set to 'mat', loads the models from the config file
        Then config_file must be specified

        if load_type is not set to 'mat'
        Then hq_mod, lq_mod, hq_params, lq_params have to be specified
        """
        self.th1 = 0.9999                # threshold for HQ clicks
        self.th2 = 0.55                  # threshold for LQ clicks
        self.lowcutfreq = 100e3          # Lowcut frequency 
        self.highcutfreq = 160e3         # Highcut frequency

        if load_type == 'mat':
            self.models_from_mat(kwargs['config_file'])
        else:
            for key, val in kwargs.items(): 
                self.__dict__[key] = val
    

    def models_from_config(self, config_file):
        """
        Read the models from the config files 
        """
        config = configparser.ConfigParser()
        config.read(configfile_path)

        logitCoefHQ = np.array(config['MODEL']['logitCoefHQ'])
        logitCoefLQ = np.array(config['MODEL']['logitCoefLQ'])
        
        self.hq_params = np.array(config['MODEL']['hq_params'])
        self.lq_params = np.array(config['MODEL']['lq_params'])


    
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
        df['pyPorCC'] = 0
        # Evaluate the model on the given x
        df['prob_hq'] = self.hq_mod.predict_proba(df[self.hq_params])[:, 1]
        df['prob_lq'] = self.lq_mod.predict_proba(df[self.lq_params])[:, 1]

        # Decide
        loc_idx = (df['CF'] > self.lowcutfreq) & (df['CF'] < self.highcutfreq) & (df['Q'] > 4)
        df.loc[~loc_idx, 'pyPorCC'] = 3                                                                      # N Clicks
        df.loc[loc_idx & (df['prob_hq'] > self.th1), 'pyPorCC'] = 1                                          # HQ Clicks
        df.loc[loc_idx & (df['prob_hq'] < self.th1) & (df['prob_lq'] > self.th2), 'pyPorCC'] = 2             # LQ Clicks
        df.loc[loc_idx & (df['prob_hq'] < self.th1) & (df['prob_lq'] <= self.th2), 'pyPorCC'] = 3            # N Clicks

        return df

    
    def test_classification_vs_matlab(self, test_df):
        """
        Test the algorithm. With the same parameters, test the prediction output of the algorithm
        """
        predicted_df = self.classify_matrix(test_df)

        # Compare 'pyPorCC' vs 'ManualAsign'
        error = np.sum(test_df['ClassifiedAs'] != predicted_df['pyPorCC'])/len(test_df)

        return error, predicted_df


    def test_PorCC(self, X, y):
        """
        Test the algorithm 
        """
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