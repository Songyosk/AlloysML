"""
Module to perform permutation importance

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  

import os                                                                                                                                                                                  
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


class permutation_importance_of_features():
    """
    Class to perform permutation importance analysis

    args: 
        (1) path_to_file (type:str) - location of the data file with features
        (2) path_to_save (type:str) - location to save new data files
        (3) target (type:str) - name of target variable
        (4) features (list) - list of exploratory features (e.g. those with multicollinearity reduced)
        (5) scaled (bool) - whether the features are scaled in the training dataset
        (5) problem (type:str) - whether it is a 'classification' or 'regression' problem
    
    return: 
        (1) list of features obtained by applying RFE
    """

    def __init__(self, path_to_file, path_to_save, target, features, scaled, problem, *args, **kwargs):
        self.path_to_save = path_to_save
        self.sample_train = joblib.load(path_to_file) 

        # Define input and target variables
        if isinstance(features, list):
            self.features = features
        else:
            self.features = joblib.load(features) 

        self.target = target

        self.problem = problem

        print('Target:', self.target)
        print('No. of features:', len(self.features))

        if scaled is False:
            # Scale the features
            scaling = MinMaxScaler(feature_range=(0, 1))

            self.sample_train[self.features] = pd.DataFrame(
                                                scaling.fit_transform(self.sample_train[self.features].values),
                                                columns=self.sample_train[self.features].columns,
                                                index=self.sample_train[self.features].index
                                            )



    def base_model(self, boosting_method, *args, **kwargs):
        """
        Select the baseline model 

        Note: 
        For classification, multi-class models are defined as shown below
        This can be changed into a binary problem by changing the 'objective' to 'binary' for LGBMClassifier, or to 'binary:logistic' or 'binary:logitraw' for XGBClassifier (see description in links below)
        
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        https://xgboost.readthedocs.io/en/latest/parameter.html
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html

        args: 
            (1) boosting_method (type:str) - either 'lightGBM' or 'XGBoost'
            (2) objective (type:str) - For classification,'binary', 'multiclass', 'multi:softprob'

        return: 
            (1)baseline model
        """
        objective = kwargs.get('objective')

        if self.problem == 'classification':
            if boosting_method == 'lightGBM':
                self.estimator = LGBMClassifier(
                                                boosting_type='gbdt',
                                                objective=objective,
                                                importance_type='gain',
                                                max_depth=-1,
                                                verbose=-1
                                                )

            elif boosting_method == 'XGBoost':
                    self.estimator = XGBClassifier(
                                                    objective=objective,
                                                    booster='gbtree',
                                                    importance_type='total_gain'
                                                    )

        elif self.problem == 'regression':
            if boosting_method == 'lightGBM':
                self.estimator = LGBMRegressor(
                                                boosting_type ='gbdt',
                                                importance_type='gain',
                                                max_depth=-1,
                                                verbose=-1
                                                )

            elif boosting_method == 'XGBoost':
                self.estimator = XGBClassifier(
                                                objective='reg:squarederror',
                                                booster='gbtree',
                                                random_state=42,
                                                importance_type='total_gain'
                                                )

        return self.estimator



    def perform(self, cv_fold=10, save=True):
        """
        Perform RFE
        """

        # Define metric to use
        if self.problem == 'classification':
            self.scoring = 'f1_weighted'

        elif self.problem == 'regression':
            self.scoring = 'neg_root_mean_squared_error'

        # Fit estimator to training data 
        self.estimator = self.estimator.fit(self.sample_train[self.features],self.sample_train[self.target].values.ravel())

        self.result = permutation_importance(
                                        estimator = self.estimator, 
                                        X = self.sample_train[self.features], 
                                        y = self.sample_train[self.target].values.ravel(),
                                        scoring = self.scoring,
                                        n_repeats = cv_fold, 
                                        random_state = 42
                                        )
        
        
        if save: 
            joblib.dump(self.result, self.path_to_save + 'permutation_importance.pkl')
            
            print('Saved as ' + str(self.path_to_save) + 'permutation_importance.pkl')
            
        return self.result
    
    
    
    def plot(self, top_n=5, *args, **kwargs):
        
        # Custom list of x
        x_list = kwargs.get('x_list')
        x_unit = kwargs.get('x_unit')
        tight_plot = kwargs.get('tight_plot')

        perm_sorted_idx = self.result.importances_mean.argsort()[-top_n:]
        tree_importance_sorted_idx = np.argsort(self.estimator.feature_importances_)[-top_n:]
        tree_indices = np.arange(0, len(self.estimator.feature_importances_))[:top_n] + 0.5

        fontsize = 10
        
        if x_list is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))           
            ax1.barh(tree_indices, self.estimator.feature_importances_[tree_importance_sorted_idx], height=0.7)
            ax1.set_yticks(tree_indices)
            ax1.set_yticklabels(np.array(self.features)[tree_importance_sorted_idx], fontsize = fontsize)
            ax1.set_ylim((0, top_n))
            ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
            ax2.boxplot(
                self.result.importances[perm_sorted_idx].T,
                vert=False,
                labels=np.array(self.features)[perm_sorted_idx],
            )
            
            
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
            ax1.barh(tree_indices, self.estimator.feature_importances_[tree_importance_sorted_idx], height=0.7)
            ax1.set_yticks(tree_indices)
            ax1.set_yticklabels(np.array(x_list)[tree_importance_sorted_idx], fontsize = fontsize)
            ax1.set_ylim((0, top_n))
            ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
            ax2.boxplot(
                self.result.importances[perm_sorted_idx].T,
                vert=False,
                labels=np.array(x_list)[perm_sorted_idx],
            )
            
        fontsize = 25

        ax1.set_ylabel('Feature number in order of their relevance', fontsize=fontsize)
        ax1.set_xlabel('Total loss reduction', fontsize=fontsize)    
        
        ax2.set_ylabel('Feature number in order of \n permutation importance', fontsize=fontsize)
        
        
        # Define metric to use
        if self.problem == 'classification':
            ax2.set_xlabel('Reduction in F1-score', fontsize=fontsize)   

        elif self.problem == 'regression':
            if x_unit is not None:
                ax2.set_xlabel('Increase in RMSE (' + str(x_unit) + ')', fontsize=fontsize)   
            else:
                ax2.set_xlabel('Increase in RMSE ' , fontsize=fontsize)   
        
        ax1.tick_params(axis='both', which='major', labelsize=fontsize, direction = 'in')
        ax2.tick_params(axis='both', which='major', labelsize=fontsize, direction = 'in')
        
        if tight_plot:   
            fig.tight_layout()
            
        plt.show()
            
        
        fig.savefig(os.path.join(self.path_to_save, r'permutation_importance_plot_' + self.target + '.png'), dpi = 300, bbox_inches="tight")

        print('Figure saved as: permutation_importance_plot_' + self.target + '.png')
        
        return np.array(self.features)[perm_sorted_idx][::-1], np.mean(self.result.importances[perm_sorted_idx].T, axis=0)[::-1]
