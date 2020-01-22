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

import sound_click



class ClickModel:
    def __init__(self, click_model_path, fs, train_hq_df, train_lq_df, test_df):
        """
        Find the click model 
        """
        self.ind_vars = ['Q', 'duration', 'ratio', 'XC', 'CF', 'BW']
        self.dep_vars = ['P']
        self.click_model = sf.read(click_model)
        self.fs = fs
        self.train_hq_df = train_hq_df
        self.train_lq_df = train_lq_df
        self.test_df = test_df


    def pick_df(self, name):
        """
        Pick the right df
        """
        if name == 'HQ':
            df = self.train_hq_df
        elif name == 'LQ':
            df = self.train_lq_df
        elif name == 'Test':
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
        y = df['P']
        X = df[comb]

        # Create the logit model
        logreg = LogisticRegression()
        logreg.fit(X, y)

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
                print(comb)
                # Regression model
                y = df['P']
                X = df[list(comb)]
                logit_model = sm.genmod.GLM(endog=y, exog=X, family=sm.families.family.Binomial())
                reg = logit_model.fit()
                
                # reg = linear_model.LogisticRegression(max_iter=1000)
                # reg.fit(df[list(comb)], df['P'])
                # models.append([comb, reg, reg.score(df[list(comb)], df['P'])])
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


def test_click_calculation(self, save=None):
    """
    Test the calculation of the click parameters obtained with python compared to the ones obtained in the paper (on the Test DB)
    """
    df_test = self.pick_df('Test')

    df_clicks = calculate_clicks_params(df_test)
    df = df_clicks.join(df_test, lsuffix='_mat', rsuffix='_py')
    if save:
        df.to_csv('data/calculated_clicks.csv')

    # Compare each field
    error = self.Test[self.ind_vars] - df_clicks[self.ind_vars]
    
    return error, df



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
        if (X['PF'] > self.lowcutfreq) and (X['PF'] < self.highcutfreq) and (X['CF'] > self.lowcutfreq) and (X['CF'] < self.highcutfreq) and (X['Q'] > 4):
            # Evaluate the model on the given x
            prob_hq = self.hq_mod.predict(X[self.hq_params])

            # Assign clip to a category
            if prob_hq >= self.th1:
                # HQ click
                porps = 0  
            else:
                prob_lq = self.lq_mod.predict(X[self.lq_params])
                if prob_lq > self.th2:
                    # LQ click
                    porps = 1  
                else:
                    # HF Noise
                    porps = 2  
        else:
            porps = 2

        return porps        


    def classify_matrix(self, df):
        """
        Classify according to the params [PF, CF, Q, XC, duration, ratio, BW]
        """
        df['pyPorCC'] = 0
        # Evaluate the model on the given x
        df['ProbHQ'] = self.hq_mod.predict_proba(df[self.hq_params])[:, 1]
        df['ProbLQ'] = self.lq_mod.predict_proba(df[self.lq_params])[:, 1]

        # Decide
        loc_idx = (df['CF'] > self.lowcutfreq) & (df['CF'] < self.highcutfreq) & (df['Q'] > 4)
        df.loc[~loc_idx, 'pyPorCC'] = 2                                                                   # N Clicks
        df.loc[loc_idx & (df['ProbHQ'] > self.th1), 'pyPorCC'] = 0                                         # HQ Clicks
        df.loc[loc_idx & (df['ProbHQ'] < self.th1) & (df['ProbLQ'] > self.th2), 'pyPorCC'] = 1             # LQ Clicks
        df.loc[loc_idx & (df['ProbHQ'] < self.th1) & (df['ProbLQ'] <= self.th2), 'pyPorCC'] = 2            # N Clicks

        return df

    
    def test_classification_vs_matlab(self, test_df):
        """
        Test the algorithm. With the same parameters, test the prediction output of the algorithm
        """
        predicted_df = self.classify_matrix(test_df)

        # Compare 'pyPorCC' vs 'ManualAsign'
        error = test_df['ClassifiedAs'] - predicted_df['pyPorCC']

        return predicted_df


    def test_PorCC(self, X, y):
        """
        Test the algorithm 
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