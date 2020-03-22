import numpy as np
import matplotlib.pyplot as plt 
import os
import datetime as dt
import itertools
import pandas as pd
from scipy import signal as sig
import scipy.stats as stats
import statsmodels.api as sm
from sklearn import metrics, linear_model, ensemble, svm, neighbors, pipeline, preprocessing, feature_selection, model_selection, utils
import pickle



class PorpoiseClassifier:
    def __init__(self, train_data=None, test_data=None, ind_vars=None, dep_var=None):
        """
        Find the click model 
        """
        if ind_vars is None:
            self.ind_vars = ['Q', 'duration', 'ratio', 'XC', 'CF', 'BW']
        else:
            self.ind_vars = ind_vars
        if dep_var is None: 
            self.dep_var = 'class'
        else:
            self.dep_var = dep_var
        self.train_data = train_data
        self.test_data = test_data
        self.models = {}


    def prepare_train_data(self, train_df):
        """
        Change the labels to convert it to a binary problem but keep the HQ/LQ information 
        """
        train_df['class'] = train_df['P'].replace(2, 1)
        train_df['class'] = train_df['class'].replace(3, 0)
        self.train_data = train_df


    def prepare_test_data(self, test_df):
        """
        Change the labels to convert it to a binary problem but keep the HQ/LQ information
        """
        test_df['class'] = test_df['ManualAsign'].replace(2, 1)
        test_df['class'] = test_df['class'].replace(3, 0)
        self.test_data = test_df


    def join_train_data(self, train_hq_df, train_lq_df):
        """
        Join the two training datasets to one to perform multiclass classification
        """
        train_hq_df['P'] = train_hq_df['P'].replace(0, 3)
        train_lq_df['P'] = train_lq_df['P'].replace(0, 3)
        train_lq_df['P'] = train_lq_df['P'].replace(1, 2)

        # Join the two datasets
        train_df = train_hq_df.append(train_lq_df, ignore_index=True)

        return train_df

    
    def join_and_split_data(self, train_hq_df, train_lq_df, test_df):
        """
        Join all the data sets and get a random separation of train/test
        """
        df = pd.DataFrame()
        train_hq_df[self.dep_var] = train_hq_df[self.dep_var].replace(0, 3)
        train_lq_df[self.dep_var] = train_lq_df[self.dep_var].replace(0, 3)
        train_lq_df[self.dep_var] = train_lq_df[self.dep_var].replace(1, 2)
        test_df.rename({'ManualAsign': 'P'})  

        df = train_hq_df[self.dep_var].append(train_lq_df[self.dep_var], ignore_index=True)
        df = df.append(test_df[self.ind_vars + [self.dep_var]], ignore_index=True)
        df['class'] = df['P'].replace(2, 1)
        df['class'] = df['class'].replace(3, 0)

        self.train_data, self.test_data = model_selection.train_test_split(df, train_size=0.4)

        return train_data, test_data

    
    def get_best_model(self, model_name, standarize=False, feature_sel=False):
        """
        Train all the classifiers. The implemented ones are
        `svc`: Support Vector Machines 
        `lsvc`: Linear Support Vector Machines
        `RandomForest`: Random Forest
        `knn`: K-Nearest Neighbor

        """
        X = self.train_data[self.ind_vars]
        y = self.train_data[self.dep_var]

        # If standarize is considered, append it to the pipeline steps
        steps = []
        if standarize: 
            # Standarize the data
            scaler = preprocessing.StandardScaler()
            steps.append(('scaler', scaler))
        
        # Some common parameters
        tol = 1e-4
        gamma = utils.fixes.loguniform(1e-4, 1000)
        C_values = utils.fixes.loguniform(0.1, 1000)
        class_weight = ['balanced', None]

        # Get the model
        if model_name == 'svc':
            # List all the possible parameters that want to be checked
            kernel_list = ['poly', 'rbf']
            degree = stats.randint(1,4)
            param_distr = {'kernel':kernel_list, 'degree':degree, 'C':C_values, 'gamma':gamma, 'class_weight':class_weight}
            # Classifier with fixed values
            clf = svm.SVC(tol=tol, cache_size=500, probability=True)

        elif model_name == 'logit':
            # List all the possible parameters that want to be checked
            penalty = ['l1', 'l2', 'elasticnet', 'none']
            param_distr = {'penalty':penalty, 'C':C_values, 'class_weight':class_weight}
            # Classifier with fixed values
            clf = linear_model.LogisticRegression()

        elif model_name == 'forest':
            # List all the possible parameters that want to be checked
            n_estimators = stats.randint(100, 300)
            param_distr = {'n_estimators': n_estimators}
            # Classifier with fixed values
            clf = ensemble.RandomForestClassifier()

        elif model_name == 'knn':
            # List all the possible parameters that want to be checked
            n_neighbors = stats.randint(2, 9)
            algorithm = ['auto', 'ball_tree', 'kd_tree']
            param_distr = {'n_neighbors':n_neighbors, 'algorithm':algorithm}
            # Classifier with fixed values
            clf = neighbors.KNeighborsClassifier()

        else: 
            raise Exception('%s is not implemented!' % (model_name))

        if feature_sel:
            # selection = feature_selection.RFECV(estimator=svm.LinearSVC(), step=1, scoring='roc_auc')
            selection = feature_selection.SelectFromModel(ensemble.ExtraTreesClassifier(n_estimators=50))
            # selection = feature_selection.SelectFromModel(svm.LinearSVC())
            # Add the feature selection to the steps
            steps.append(('feature_selection', selection))

        # Search for the best parameters
        gm_cv = model_selection.RandomizedSearchCV(estimator=clf, scoring='roc_auc', param_distributions=param_distr)
        steps.append(('classification', gm_cv))

        # Create pipeline and fit
        model = pipeline.Pipeline(steps)
        model.fit(X, y)
        self.models[model_name] = model

        return self.models[model_name]

    
    def train_models(self, model_list, standarize=False, feature_sel=False, verbose=True):
        """
        Train all the models of the list
        """
        results = pd.DataFrame(columns=['name', 'roc_auc', 'recall', 'aic'])
        for model_name in model_list: 
            self.get_best_model(model_name, standarize, feature_sel)
            
            # Plot the results
            y_test = self.test_data[self.dep_var]
            y_pred = self.models[model_name].predict(self.test_data[self.ind_vars])
            y_prob = self.models[model_name].predict_proba(self.test_data[self.ind_vars])
            if feature_sel:
                n_features = self.models[model_name]['feature_selection'].transform(self.test_data[self.ind_vars]).shape[1]
            else: 
                n_features = self.test_data[self.ind_vars].shape[1]

            print(self.models[model_name]['classification'].best_estimator_)
            print(metrics.classification_report(y_test, y_pred))

            # Calculate rou_aub, recall and aic
            roc_aub = metrics.roc_auc_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            aic = aic_score(y_test, y_prob, n_features)
            results.loc[results.size] = [model_name, roc_aub, recall, aic]
        
        print(results)

    

    def train_cnn(self):
        """
        Train a CNN 
        """
        return 0


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

        return      


    def classify_matrix(self, df):
        """
        Classify according to the params 
        """

        return df

    
    def test_classification_vs_matlab(self, test_df):
        """
        Test the algorithm. With the same parameters, test the prediction output of the algorithm
        """
        predicted_df = self.classify_matrix(test_df)

        error = np.sum(test_df['ClassifiedAs'] != predicted_df['pyPorCC'])/len(test_df)

        return error, predicted_df


def aic_score(y, y_prob, n_features):
    """
    Return the AIC score
    """
    llk = np.sum(y*np.log(y_prob[:, 1]) + (1 - y)*np.log(y_prob[:,0]))
    aic = 2*n_features - 2*(llk)
    return aic