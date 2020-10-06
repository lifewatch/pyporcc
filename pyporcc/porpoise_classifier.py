#!/usr/bin/python
"""
Module : porpoise_classifier.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Instituut voor de Zee)
Last Accessed : 9/23/2020
"""

from pyporcc import utils

import pickle
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn import metrics, linear_model, ensemble, svm, neighbors, pipeline
from sklearn import preprocessing, feature_selection, model_selection, utils


class PorpoiseClassifier:
    def __init__(self, train_data=None, test_data=None, ind_vars=None, dep_var=None):
        """
        Find the click model

        Parameters
        ----------
        train_data : DataFrame
            Data to train.
        test_data : DataFrame
            Data to test
        ind_vars : list or np.array
            List of the independent variables to consider (and names of the columns)
        dep_var : string
            Name of the dependent variable (name of the column)
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

    def split_data(self, data_df, train_size=0.4):
        """
        Split the data in train and test and save it to the appropriate attributes of the class

        Parameters
        ----------
        data_df : DataFrame
            DataFrame with all the data to be divided
        train_size : float
            Float between 0 and 1 representing the percentage of training data
        """
        self.train_data, self.test_data = model_selection.train_test_split(data_df, train_size=train_size)
        return self.train_data, self.test_data

    def get_best_model(self, model_name, binary=False, standarize=False, feature_sel=False):
        """
        Train all the classifiers.

        Parameters
        ----------
        model_name : string
            Name of the model to test.
            The implemented ones are
            `svc`: Support Vector Machines
            `lsvc`: Linear Support Vector Machines
            `RandomForest`: Random Forest
            `knn`: K-Nearest Neighbor
        binary : bool
            If true the variables are converted to 0 / 1 (no porpoise / porpoise) instead of noise/lq/hq clicks
        standarize : bool
            Set to True if the variables should be standarized before training
        feature_sel : bool
            Set to True if the best features should be selected instead of all of them

        Returns
        -------
        Dictionary with the name of the model as key and another dictionary as value
        with ind_vars, model, binary as keys and their respective representations in values
        (It also adds it to the property "models" of the class)
        """    
        x = self.train_data[self.ind_vars]
        y = self.train_data[self.dep_var]
        if binary: 
            # Convert the classes in 0 (no porpoise) or 1 (porpoise)
            y = self.convert2binary(y)

        # If standarize is considered, append it to the pipeline steps
        steps = []
        if standarize: 
            # Standarize the data
            scaler = preprocessing.StandardScaler()
            steps.append(('scaler', scaler))
        
        # Some common parameters
        tol = 1e-3
        gamma = utils.fixes.loguniform(1e-4, 1000)
        c_values = utils.fixes.loguniform(0.1, 1000)
        class_weight = ['balanced', None]

        # Get the model
        if model_name == 'svc':
            # List all the possible parameters that want to be checked
            kernel_list = ['poly', 'rbf'] 
            degree = stats.randint(1, 4)
            param_distr = {'degree': degree, 'C': c_values, 'gamma': gamma, 'kernel': kernel_list}
            
            # Classifier with fixed values
            clf = svm.SVC(tol=tol, cache_size=500, probability=True, max_iter=500)

        elif model_name == 'logit':
            # List all the possible parameters that want to be checked
            penalty = ['l1', 'l2', 'elasticnet', 'none']
            param_distr = {'penalty': penalty, 'C': c_values, 'class_weight': class_weight}
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
            param_distr = {'n_neighbors': n_neighbors, 'algorithm': algorithm}
            # Classifier with fixed values
            clf = neighbors.KNeighborsClassifier()

        else: 
            raise Exception('%s is not implemented!' % model_name)

        if feature_sel:
            # selection = feature_selection.RFECV(estimator=svm.LinearSVC(), step=1, scoring='roc_auc')
            selection = feature_selection.SelectFromModel(ensemble.ExtraTreesClassifier(n_estimators=50))
            # selection = feature_selection.SelectFromModel(svm.LinearSVC())
            # Add the feature selection to the steps
            steps.append(('feature_selection', selection))

        # Search for the best parameters
        gm_cv = model_selection.RandomizedSearchCV(estimator=clf, scoring='roc_auc',
                                                   param_distributions=param_distr, n_iter=100)
        steps.append(('classification', gm_cv))

        # Create pipeline and fit
        model = pipeline.Pipeline(steps)
        model.fit(x, y)

        if feature_sel:
            ind_vars = model['feature_selection'].transform(self.test_data[self.ind_vars])
        else: 
            ind_vars = self.ind_vars

        print(model['classification'].best_estimator_)
        self.models[model_name] = {'ind_vars': ind_vars, 'model': model, 'binary': binary}
        
        # Save the model as a pickle file! 
        pickle.dump(model, open('pyporcc/models/%s.pkl' % model_name, 'wb'))
        return self.models[model_name]

    def train_models(self, model_list, binary=False, standarize=False, feature_sel=False):
        """
        Train all the models of the list

        Parameters
        ----------
        model_list : list of strings
            All the models that want to be tested. They have to be the implemented types (see get_best_model)
        binary : bool
            If true the variables are converted to 0 / 1 (no porpoise / porpoise) instead of noise/lq/hq clicks
        standarize : bool
            Set to True if the variables should be standarized before training
        feature_sel : bool
            Set to True if the best features should be selected instead of all of them

        Returns
        -------
        Dictionary with the name of each of the models of the list as keys and another dictionary as value
        with ind_vars, model, binary as keys and their respective representations in values
        (It also adds it to the property "models" of the class)
        """
        for model_name in model_list: 
            self.get_best_model(model_name, binary, standarize, feature_sel)
        return self.models

    def test_models(self):
        """
        Test the models

        Returns
        -------
        DataFrame with name, roc_auc, recall and aic as columns
        """
        results = pd.DataFrame(columns=['name', 'roc_auc', 'recall', 'aic'])
        for model_name, model_item in self.models.items(): 
            ind_vars = model_item['ind_vars']     
            model = model_item['model']       
            y_test = self.test_data[self.dep_var]
            if model_item['binary']:
                # Convert the classes in 0 (no porpoise) or 1 (porpoise)
                y_test = self.convert2binary(y_test)
            y_pred = model.predict(self.test_data[ind_vars])
            y_prob = model.predict_proba(self.test_data[ind_vars])

            print(metrics.classification_report(y_test, y_pred))

            # Calculate rou_aub, recall and aic
            roc_auc = metrics.roc_auc_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            aic = utils.aic_score(y_test, y_prob, len(ind_vars))
            results.loc[results.size] = [model_name, roc_auc, recall, aic]
        
        print(results)
        return results  

    def plot_roc_curves(self, porcc_al):
        """
        Plot the roc curves for HQ vs Noise, LQ vs Noise and All vs Noise

        Parameters
        ----------
        porcc_al : object PorCCModel
            Model to compare it to
        """
        fig, ax = plt.subplots(1, 3)
        hq_noise = self.test_data.loc[self.test_data[self.dep_var] != 2]
        lq_noise = self.test_data.loc[self.test_data[self.dep_var] != 1]
        self._plot_roc_curves(porcc_al=porcc_al, df=hq_noise, ax=ax[0])
        ax[0].set_title('HQ vs Noise')
        self._plot_roc_curves(porcc_al=porcc_al, df=lq_noise, ax=ax[1])
        ax[1].set_title('LQ vs Noise')
        self._plot_roc_curves(porcc_al=porcc_al, df=self.test_data, ax=ax[2])
        ax[2].set_title('All vs Noise')

        plt.tight_layout()
        plt.show()
        plt.close()

    def _plot_roc_curves(self, porcc_al, df, ax):
        """
        Plot the roc curves of all the models

        Parameters
        ----------
        porcc_al : object PorCCModel
            Model to compare it to
        df : DataFrame
            DataFrame to consider
        ax : matplotlib ax
            Axis where to plot the ROC curves
        """
        y_test = df[self.dep_var]
        for model_name, model_item in self.models.items():
            ind_vars = model_item['ind_vars']     
            model = model_item['model']  
            x_test = df[ind_vars]
            if model_item['binary']:
                # Convert the classes in 0 (no porpoise) or 1 (porpoise)
                y_test = self.convert2binary(y_test)
            y_prob = model.predict_proba(x_test)[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, label=model_name)
        
        porcc_al._plot_roc_curves(df, dep_var=self.dep_var, ax0=ax)
        ax.set_xlabel('False alarm rate')
        ax.set_ylabel('Hit rate')
        ax.legend()
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

    def add_model(self, model_name, model, ind_vars, binary=False):
        """
        Add a model to the models dict

        Parameters
        ----------
        model_name : string
            Name of the new model
        model : object
            Model object already fitted
        ind_vars : list or np.array of strings
            Names of the columns of the independent variables
        binary : bool
            Set to True if the dependent variable should be converted to a binary class
        """
        self.models[model_name] = {'ind_vars': ind_vars, 'model': model, 'binary': binary}

    def save(self, path):
        """
        Save the current models in a file. Should be a pickle extension

        Parameters
        ----------
        path : string or Path
            Where to store the self object
        """
        extension = path.split('.')[-1]
        if extension == 'pkl':
            pickle.dump(self, open(path, 'wb'))
        else:
            raise Exception('The %s extension is unknown!' % extension)

    def classify_click(self, click):
        """
        Classify the click in HQ, LQ, N

        Parameters
        ----------
        click : Click object
            Click to classify

        Returns
        -------
        Class predictions for each of the models calculated
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
        x : DataFrame row
            Needs to have the independent variables needed for all the trained models

        Returns
        -------
        A DataFrame with the class prediction for all the trained models
        """
        for model_info in self.models:
            x['model'] = model_info['model'].predict(x[model_info['ind_vars']])
        return x

    def classify_matrix(self, df):
        """
        Classify according to the params

        Parameters
        ----------
        df : DataFrame
            DataFrame to classify

        Returns
        -------
        A DataFrame with the class prediction for all the trained models
        """
        for model_info in self.models:
            df['model'] = model_info['model'].predict(df[model_info['ind_vars']])
        return df
