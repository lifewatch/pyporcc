import os
import pickle
import itertools
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt 


from scipy import signal as sig
from sklearn import metrics, linear_model, ensemble, svm, neighbors, pipeline, preprocessing, feature_selection, model_selection, utils



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


    def split_data(self, data_df, train_size=0.4):
        """
        Prepare train, test
        """
        self.train_data, self.test_data = model_selection.train_test_split(data_df, train_size=train_size)

        return self.train_data, self.test_data
    

    def split_2trainingsets(self, data_df, train_size=0.2):
        """
        Prepare two training sets, one with HQ and the other one with LQ
        """
        # TO BE IMPLEMENTED!
        return 0

    
    def get_best_model(self, model_name, binary=False, standarize=False, feature_sel=False):
        """
        Train all the classifiers. The implemented ones are
        `svc`: Support Vector Machines 
        `lsvc`: Linear Support Vector Machines
        `RandomForest`: Random Forest
        `knn`: K-Nearest Neighbor

        binary : converts the variables to 0 / 1 (no porpoise / porpoise) instead of lq/hq clicks


        """    
        X = self.train_data[self.ind_vars]
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
        C_values = utils.fixes.loguniform(0.1, 1000)
        class_weight = ['balanced', None]

        # Get the model
        if model_name == 'svc':
            # List all the possible parameters that want to be checked
            kernel_list = ['poly', 'rbf'] 
            degree = stats.randint(1,4)
            param_distr = {'degree':degree, 'C':C_values, 'gamma':gamma, 'kernel':kernel_list}
            
            # Classifier with fixed values
            clf = svm.SVC(tol=tol, cache_size=500, probability=True, max_iter=500)

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
        gm_cv = model_selection.RandomizedSearchCV(estimator=clf, scoring='roc_auc', param_distributions=param_distr, n_iter=100)
        steps.append(('classification', gm_cv))

        # Create pipeline and fit
        model = pipeline.Pipeline(steps)
        model.fit(X, y)

        if feature_sel:
            ind_vars = model['feature_selection'].transform(self.test_data[self.ind_vars])
        else: 
            ind_vars = self.ind_vars

        print(model['classification'].best_estimator_)
        self.models[model_name] = {'ind_vars': ind_vars, 'model': model, 'binary':binary}
        
        # Save the model as a pickle file! 
        pickle.dump(model, open('pyporcc/models/%s.pkl' % (model_name), 'wb'))

        return self.models[model_name]

    
    def train_models(self, model_list, binary=False, standarize=False, feature_sel=False, verbose=True):
        """
        Train all the models of the list
        """
        for model_name in model_list: 
            self.get_best_model(model_name, binary, standarize, feature_sel)
            
        return self.models
    

    def test_models(self):
        """
        Test the models
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
            aic = aic_score(y_test, y_prob, len(ind_vars))
            results.loc[results.size] = [model_name, roc_auc, recall, aic]
        
        print(results)
        return results  


    def plot_roc_curves(self, porcc_al):
        """
        Plot the roc curves for HQ vs Noise, LQ vs Noise and All vs Noise
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
        """
        y_test = df[self.dep_var]
        for model_name, model_item in self.models.items():
            ind_vars = model_item['ind_vars']     
            model = model_item['model']  
            X_test = df[ind_vars]
            if model_item['binary']:
                # Convert the classes in 0 (no porpoise) or 1 (porpoise)
                y_test = self.convert2binary(y_test)
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, label=model_name)
        
        porcc_prob = porcc_al._plot_roc_curves(df, dep_var=self.dep_var, ax0=ax)

        ax.set_xlabel('False alarm rate')
        ax.set_ylabel('Hit rate')

        ax.legend()

        return ax


    def convert2binary(self, y):
        """
        Return the y sample as binary (no porpoise / porpoise)
        """
        y = y.replace(3, 0)
        y = y.replace(2, 1)  

        return y


    def add_model(self, model_name, model, ind_vars, binary=False):
        """
        Add a model to the models dict
        """
        self.models[model_name] = {'ind_vars':ind_vars, 'model':model, 'binary':binary}   


    def save(self, path):
        """
        Save the current models in a file. Can be chosen to save it as pickle or joblib
        """
        extension = path.split('.')[-1]
        if extension == 'pkl':
            pickle.dump(self, open(path, 'wb'))
        else:
            raise Exception('The %s extension is unknown!' % (extension))


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



def aic_score(y, y_prob, n_features):
    """
    Return the AIC score
    """
    llk = np.sum(y*np.log(y_prob[:, 1]) + (1 - y)*np.log(y_prob[:,0]))
    aic = 2*n_features - 2*(llk)
    return aic