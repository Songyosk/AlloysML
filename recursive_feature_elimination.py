"""
Module to perform recursive feature elimination

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  

import os                                                                                                                                                                                  
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFECV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


class recursive_feature_elimination():
    """
    Class to perform recursive feature elimination

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
                                                force_col_wise=True,
                                                max_depth=-1,
                                                verbose=-1
                                                )

            elif boosting_method == 'XGBoost':
                self.estimator = XGBRegressor(
                                                objective='reg:squarederror',
                                                booster='gbtree',
                                                random_state=42,
                                                importance_type='total_gain'
                                                )


        return self.estimator



    def perform(self, cv_fold=10):
        """
        Perform RFE
        """

        # Define metric to use
        if self.problem == 'classification':
            self.scoring = 'f1_weighted'

        elif self.problem == 'regression':
            self.scoring = 'neg_root_mean_squared_error'


        # Define step and cv to apply to RFECV
        self.min_features_to_select = 1
        step = 1
        cv = cv_fold

        self.selector = RFECV(
                                self.estimator, 
                                min_features_to_select = self.min_features_to_select,
                                scoring = self.scoring,
                                step = step, 
                                cv = cv, 
                                verbose = 1
                                )

        # Fit to training data
        self.selector = self.selector.fit(
                                        self.sample_train[self.features], 
                                        self.sample_train[self.target].values.ravel(),

                                        )


        # Create panda.Dataframe with newly selected features
        RFECV_importance = self.selector.ranking_
        RFECV_features = pd.DataFrame({'features': self.features, 'importance_score':RFECV_importance})
        RFECV_features = RFECV_features.sort_values(by = 'importance_score', ascending = True)
        RFECV_features = RFECV_features.reset_index(drop = True)


        # Save data
        joblib.dump(self.selector, os.path.join(self.path_to_save, r'RFECV_selector_' + self.target + '.pkl'))
        joblib.dump(self.selector.cv_results_, os.path.join(self.path_to_save, r'RFECV_features_cv_results_' + self.target + '.pkl'))

        print('CV scores saved:', 'RFECV_features_cv_results_' + self.target + '.pkl')


        # RFECV_features saved above
        self.RFE_features = RFECV_features[RFECV_features.importance_score == 1]
        self.RFE_features = self.RFE_features['features']

        joblib.dump(self.RFE_features, os.path.join(self.path_to_save, r'features_selected_from_RFE_' + self.target + '.pkl'))

        print('\n Features saved as features_selected_from_RFE_' + self.target + '.pkl')
        print('\n Number of features remaining: ', len(self.RFE_features))


        return self.RFE_features



    def RFE_plot(self):
        """
        Plot the result of the RFE
        """

        # Generate figure
        fig, ax = plt.subplots(figsize = (8, 8))

        fontsize = 18
        plt.xlabel('Number of features', fontsize = fontsize)
        plt.ylabel('Cross validation result', fontsize = fontsize)
        plt.tick_params(axis = 'both', which = 'major', labelsize = fontsize, direction = 'in')

        plt.plot(
                range(self.min_features_to_select, len(self.selector.cv_results_['mean_test_score']) + self.min_features_to_select),
                self.selector.cv_results_['mean_test_score']
                )

        plt.show()

        fig.savefig(os.path.join(self.path_to_save, r'RFE_plot_' + self.target + '.png'), dpi = 300, bbox_inches="tight")

        print('Figure saved as: RFE_plot_' + self.target + '.png')



