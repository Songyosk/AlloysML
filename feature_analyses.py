"""
Module to perform feature analyses

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  

import os                                                                                                                                                                                  
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression


class perform():
    """
    Class for perform feature analyses

    args: 
        (1) path_to_file (type:str) - location of the training set; last column taken as the target feature
        (2) path_to_save (type:str) - location to save new data files
        (3) target (type:str) - name of target variable
        (4) features (list) - list of exploratory features

    return: 
        (1) pandas.Dataframe of analysis result
    """

    def __init__(self, path_to_file, path_to_save, target, features):
        self.path_to_save = path_to_save
        self.sample_train = joblib.load(path_to_file)

        # Define input and target variables
        if isinstance(features, list):
            self.features = features
        else:
            self.features = joblib.load(features) 

        self.target = target

        print('Name of target column: ', self.target)
        print('No. of exploratory features: ', len(self.features))



    def remove_constant_features(self):
        """
        Remove features with constant values e.g. all zeros
        """

        self.sample_train = self.sample_train[self.features + [self.target]]
        
        # No. of exploratory features
        no_f_0 = len(self.sample_train.columns)

        # Set variance threshold
        variance_threshold = VarianceThreshold(threshold = 0)

        # Apply to dataset
        variance_threshold.fit_transform(self.sample_train[self.features])

        # Define new dataframe
        col = variance_threshold.get_support(indices=True).tolist()

        # Add index of target column
        col = col + [self.sample_train.columns.get_loc(self.target)]

        # Select relevant columns and redefine self.sample_train
        self.sample_train = self.sample_train.iloc[:, col] 

        # No. of exploratory features after treatment
        no_f_1 = len(self.sample_train.columns)

        print('No. of features removed: ', no_f_0 - no_f_1)
        print('No. of exploratory features: ', len(self.sample_train.columns) - 1)


        return self.sample_train



    def ANOVA_F_test(self, problem, csv, all_features=False):
        """
        Conduct ANOVA F-test: 
        (a) for classification with more than two target classees
        (b) for regression where corresponding F-statistics are computed using the correlation coeff

        args: 
            (1) problem (type:str) - specify whether it is for a 'classification' or a 'regression' problem
            (2) all_features (type:bool) - specify whether all features should be considered or just numerical features
            (3) csv (type:bool) - whether to save result in csv format

        return: 
            (1) result of ANOVA F-test
        """

        # Columns with 'ohe' strings are categorical
        self.categorical_cols = [i for i in self.sample_train.columns if '_ohe_' in i]


        # Rest are numerical
        self.numerical_cols = [i for i in self.sample_train.columns if (i not in self.categorical_cols) is True and (i != self.target) is True and (i in self.features) is True]
        

        print('There are:')
        print('No. of categorical features: ', len(self.categorical_cols))
        print('No. of numerical features: ', len(self.numerical_cols), '\n')


        # Consider all numerical features
        if problem == 'classification':
            sel_f = SelectKBest(f_classif, k='all')

        elif problem == 'regression':
            sel_f = SelectKBest(f_regression, k='all')

        if all_features:
            feature_list = self.categorical_cols + self.numerical_cols
            print('Considering all features. \n')
        else:
            feature_list = self.numerical_cols
            print('Considering only the numerical features. \n')

        df_features = self.sample_train[feature_list]
    

        # Fit to data
        df_target = self.sample_train[self.target]
        df_train_f = sel_f.fit_transform(df_features, df_target)


        # Extract indices of features considered successfully
        col = sel_f.get_support(indices=True).tolist()
        df_train_f = df_features.iloc[:, col]


        # Table of F-statistic values
        if problem == 'classification':
            names = df_train_f[feature_list].columns.values[sel_f.get_support()]

        elif problem == 'regression':
            names = df_train_f[feature_list].columns.values[sel_f.get_support()]

        scores = sel_f.scores_[sel_f.get_support()]
        df_stat = pd.DataFrame(
                                data = list(zip(names, scores)), 
                                columns = ['feature_names','f_statistic']).sort_values(['f_statistic','feature_names'], 
                                ascending = [False, True]
                                )


        # Scale F-statistic
        scaling = MinMaxScaler()

        df_stat = df_stat[df_stat['f_statistic'].notnull()]
        df_stat['f_statistic_scaled'] = scaling.fit_transform(df_stat['f_statistic'].values.reshape(-1,1))

        df_stat = df_stat.reset_index(drop=True)
        df_stat = df_stat[df_stat['feature_names'].notna()]
        df_stat = df_stat.sort_values('f_statistic_scaled', ascending=False)


        # Save results
        if csv == True:
            df_stat.to_csv(os.path.join(self.path_to_save, r'ANOVA_F_test_result_' + self.target + '_' + problem + '.csv'))

            print('Result saved as: ', 'ANOVA_F_test_result_' + self.target + '_'  + problem + '.csv')


        joblib.dump(df_stat, os.path.join(self.path_to_save, r'ANOVA_F_test_result_' + self.target + '_'  + problem + '.pkl'))

        print('Result saved as: ', 'ANOVA_F_test_result_' + self.target + '_'  + problem + '.pkl')


        return df_stat



    def chi2_test(self, csv):
        """
        Conduct Chi-squared test for categorical assoication

        args: 
            (1) csv (type:bool) - whether to save result in csv format
        
        return: 
            (1) result of Chi-sqaured test
        """

        # Columns with 'ohe' strings are categorical
        self.categorical_cols = [i for i in self.sample_train.columns if '_ohe_' in i]


        print('There are:')
        print('No. of categorical features: ', len(self.categorical_cols))


        # Consider all numerical features
        sel_c = SelectKBest(chi2, k='all')


        df_categorical = self.sample_train[self.categorical_cols]
        df_target = self.sample_train[self.target]


        # Fit to data
        df_train_c = sel_c.fit_transform(df_categorical, df_target)


        # Extract indices of features considered successfully
        col = sel_c.get_support(indices=True).tolist()
        df_train_c = df_categorical.iloc[:, col]


        # Table of Chi2 values
        names = df_train_c[self.categorical_cols].columns.values[sel_c.get_support()]
        scores = sel_c.scores_[sel_c.get_support()]


        df_stat = pd.DataFrame(
                                data = list(zip(names, scores)), 
                                columns = ['feature_names','chi_squared']).sort_values(['chi_squared','feature_names'], 
                                ascending = [False, True]
                                )


        # Scale Chi2
        scaling = MinMaxScaler()

        df_stat = df_stat[df_stat['chi_squared'].notnull()]
        df_stat['chi_squared_scaled'] = scaling.fit_transform(df_stat['chi_squared'].values.reshape(-1,1))
        df_stat = df_stat.reset_index(drop=True)
        df_stat = df_stat[df_stat['feature_names'].notna()]
        df_stat = df_stat.sort_values('chi_squared_scaled', ascending=False)


        # Save results
        if csv == True:
            df_stat.to_csv(os.path.join(self.path_to_save, r'chi_squared_test_result_' + self.target + '.csv'))

            print('Result saved as: chi_squared_test_result_' + self.target + '.csv')

        joblib.dump(df_stat, os.path.join(self.path_to_save, r'chi_squared_test_result_' + self.target + '.pkl'))

        print('Result saved as: chi_squared_test_result_' + self.target + ' .pkl')


        return df_stat



    def mutual_information(self, problem, csv):
        """
        Conduct mutual information analyses for either a classification or regression problem

        args:
            (1) problem (type:str) - specify whether it is a classification or a regression problem 
            (2) csv (type:bool) - whether to save result in csv format

        return: 
            (1) result of MI
        """

        # Columns with 'ohe' strings are categorical
        self.categorical_cols = [i for i in self.sample_train.columns if '_ohe_' in i]


        # Rest are numerical
        self.numerical_cols = [i for i in self.sample_train.columns if (i not in self.categorical_cols) is True and (i != self.target) is True and (i in self.features) is True]
        

        print('There are:')
        print('No. of categorical features: ', len(self.categorical_cols))
        print('No. of numerical features: ', len(self.numerical_cols), '\n')


        # Consider all numerical features
        if problem == 'classification':
            sel_m = SelectKBest(mutual_info_classif, k='all')

        elif problem == 'regression':
            sel_m = SelectKBest(mutual_info_regression, k='all')


        # Depends on whether some features were removed prior to this step e.g. removing constant features
        try:
            df_train = self.sample_train[self.features]

        except KeyError:
            df_train = self.sample_train[self.categorical_cols + self.numerical_cols]


        df_target = self.sample_train[self.target]


        # Fit to data
        df_train_m = sel_m.fit_transform(df_train, df_target)


        # Extract indices of features considered successfully
        col = sel_m.get_support(indices=True).tolist()

        df_train_m = df_train.iloc[:, col]


        # Table of MI 
        try:
            names = df_train_m[self.features].columns.values[sel_m.get_support()]

        except KeyError:
            names = df_train_m[self.categorical_cols + self.numerical_cols].columns.values[sel_m.get_support()]


        scores = sel_m.scores_[sel_m.get_support()]

        df_stat = pd.DataFrame(
                                data = list(zip(names, scores)), 
                                columns = ['feature_names','MI']).sort_values(['MI','feature_names'], 
                                ascending = [False, True]
                                )
        

        # Scale MI
        scaling = MinMaxScaler()

        df_stat = df_stat[df_stat['MI'].notnull()]
        df_stat['MI_scaled'] = scaling.fit_transform(df_stat['MI'].values.reshape(-1,1))

        df_stat = df_stat.reset_index(drop=True)
        df_stat = df_stat[df_stat['feature_names'].notna()]
        df_stat = df_stat.sort_values('MI_scaled', ascending=False)


        # Save results
        if csv == True:
            df_stat.to_csv(os.path.join(self.path_to_save, r'MI_result_' + self.target + '.csv'))

            print('Result saved as: MI_result_' + self.target + '.csv')

        joblib.dump(df_stat, os.path.join(self.path_to_save, r'MI_result_' + self.target + '.pkl'))

        print('Result saved as: MI_result_' + self.target + '.pkl')


        return df_stat



    def logistic_discrimination(self, target_classes, class_names, csv):
        """
        Perform logistic_discrimination on a classification problem.
        Note: multi_class can be ‘auto’, which selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.

        args:
            (1) target_classes (type:int) - Number of target classes
            (2) class_names (type:list) - list of target classes
            (2) csv (type:bool) - whether to save result in csv format

        return: 
            (1) result of logistic discrimination
        """

        # Columns with 'ohe' strings are categorical
        self.categorical_cols = [i for i in self.sample_train.columns if '_ohe_' in i]


        # Rest are numerical
        self.numerical_cols = [i for i in self.sample_train.columns if (i not in self.categorical_cols) is True and (i != self.target) is True]


        if target_classes == 2:
            # Binomial LR
            logreg = LogisticRegression(multi_class='auto', max_iter=300, solver='liblinear')

        elif target_classes > 2:
            # Multinomial LR
            logreg = LogisticRegression(multi_class='auto', max_iter=300, solver='lbfgs')


        # Depends on whether some features were removed prior to this step e.g. removing constant features
        try:
            df_train = self.sample_train[self.features]

        except KeyError:
            df_train = self.sample_train[self.categorical_cols + self.numerical_cols]


        df_target = self.sample_train[self.target]


        # Fit to data
        model = logreg.fit(df_train, df_target.values.ravel())


        if target_classes == 2:
            multivariate_ranking = pd.DataFrame(data = list(zip(self.features, model.coef_[0])), columns = ['feature_names', 'coefficient'])

        elif target_classes > 2:

            class_names = ['feature_names'] + class_names
            coefficent = []
            
            for i in range(target_classes):
                coefficent.append(model.coef_[i])

            multivariate_ranking = pd.DataFrame(data = list(zip(self.features, *coefficent)), columns = class_names)


        multivariate_ranking = multivariate_ranking[multivariate_ranking['feature_names'].notna()]


        # Save results
        if csv == True:
            multivariate_ranking.to_csv(os.path.join(self.path_to_save, r'logistic_discrimination_result_' + self.target + '.csv'))

            print('Result saved as: logistic_discrimination_result_' + self.target + '.pkl')

        joblib.dump(multivariate_ranking, os.path.join(self.path_to_save, r'logistic_discrimination_result_' + self.target + '.pkl'))

        print('Result saved as: logistic_discrimination_result_' + self.target + '.pkl')


        return multivariate_ranking


