"""
Module for feature engineering

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  

import os 
import numpy as np                                                                                                                                                                                 
import pandas as pd
import joblib
import itertools

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class engineering():
    """
    Class is used to engineer use features
    Two methods present: (a) brute force method, (b) feature marker

    Note: this should be used before scaling the features and before oversampling.

    args: 
        (1) path_to_file (type:str) - location of the data file with features; last column is taken as the target feature
        (2) path_to_save (type:str) - loacation to save data
        (3) target (type:str) - name of target variable
        (4) features (list) - list of exploratory features
        (5) csv (type:bool) - whether to save as csv

    return: 
        (1) pandas.Dataframe with newly engineered features
    """

    def __init__(self, path_to_file, path_to_save, target, features, csv = False):
        self.path_to_save = path_to_save
        self.csv = csv
        
        self.sample_train = joblib.load(path_to_file)

        # Define input and target variables
        if isinstance(features, list):
            self.features = features
        else:
            self.features = joblib.load(features) 

        self.target = target

        print('Name of target column: ', self.target)
        print('No. of exploratory features: ', len(self.features))



    def movecol(self, dataframe, cols_to_move = [], ref_col = '', place = 'after'):
        """
        Function to rearrange columns

        arg: 
            (a) cols_to_move (list) - list of columns to move
            (b) ref_col (type:str) - reference column 
            (c) place (type:str) - whether to move the specified columns 'before' or 'after' the reference column (set to 'after' as default)

        return:
            (a) pandas.Dataframe
        """

        cols = dataframe.columns.tolist()


        if place == 'after':
            s1 = cols[:list(cols).index(ref_col) + 1]
            s2 = cols_to_move


        if place == 'before':
            s1 = cols[:list(cols).index(ref_col)]
            s2 = cols_to_move + [ref_col]
        

        s1 = [i for i in s1 if i not in s2]
        s3 = [i for i in cols if i not in s1 + s2]
        

        return dataframe[s1 + s2 + s3]



    def brute_force(self, feature_list):
        """
        Feature engineering using brute force method
        Use features identified to have statistical significance 
        
        args: 
            (1) feature_list (list) - list of features to use

        return: 
            (1) pandas.Dataframe with newly engineered features
        """

        # List of all permutations
        all_perm = list()

        # Find possible permutations between a pair of features
        for i in itertools.permutations(set(feature_list), r=2):
            all_perm.append(list(i))

        print('Total number of permutation:', len(all_perm))
        
        # Original columns
        original_cols = self.sample_train.columns.tolist()

        self.sample_train.replace({False: 0, True: 1}, inplace=True)

        # Compute the feature values of the new features
        for f in all_perm:
            if str(f[0]) != str(f[1]):
                self.sample_train[str(f[0]) + '/' + str(f[1])] = self.sample_train[f[0]] / self.sample_train[f[1]]
            
        print('Invalid operations:', len(all_perm) - (len(self.sample_train.columns.tolist())-len(original_cols)))

        # Move target to last column
        self.sample_train = self.movecol(
                                        self.sample_train, 
                                        cols_to_move = [self.target], 
                                        ref_col = self.sample_train.columns.tolist()[-1], 
                                        place = 'after'
                                        )


        # New columns
        latest_cols = self.sample_train.columns.tolist()
        
        self.new_cols = []

        for c in latest_cols:
            if c not in original_cols:
                self.new_cols.append(c)


        # Replace str to Nan
        self.sample_train[self.new_cols] = self.sample_train[self.new_cols].applymap(lambda x: x if not isinstance(x, str) else np.nan)


        # Replace NaN with zeros
        self.sample_train = self.sample_train.fillna(0)


        # Replace the infinites to zeros
        self.sample_train = self.sample_train.replace([np.inf, -np.inf], 0)


        return self.sample_train, self.new_cols



    def feature_markers(self, feature_list):
        """
        Create feature markers (1 or 0) to indicate the presence or absence of a feature, respectively
        Use features identified to have statistical significance 
            
        args: 
            (1) feature_list (list) - list of features to use

        return: 
            (1) pandas.Dataframe with newly engineered features
        """

        # Compute the feature values of the new features
        for f in feature_list:
            self.sample_train[str(f) + '_marker'] = self.sample_train[f].apply(lambda x: 1 if x != 0 else 0)


        # Move target to last column
        self.sample_train = self.movecol(self.sample_train, cols_to_move=[self.target], ref_col=self.sample_train.columns.tolist()[-1], place='after')


        return self.sample_train



    def save(self, test_size=0.2, random_state=42):
        """
        Save data file with new features 
        """

        #Save data as csv
        if self.csv == True:
            self.sample_train.to_csv(os.path.join(self.path_to_save, r'engineered_features_' + self.target + '.csv'))

            print('Result saved as: engineered_features_' + self.target + '.csv')
        

        joblib.dump(self.sample_train, os.path.join(self.path_to_save, r'df_' + self.target + '_engineered_features.pkl'))

        print('Result saved as: df_' + self.target + '_engineered_features.pkl')


        joblib.dump(self.new_cols, os.path.join(self.path_to_save, r'features_' + self.target + '_engineered.pkl'))

        print('Result saved as: features_' + self.target + '_engineered.pkl')


        # Split dataset
        df_train, df_test = train_test_split(self.sample_train, test_size=test_size, random_state=random_state) 
        
        joblib.dump(df_train, os.path.join(self.path_to_save, r'df_train_' + self.target + '_engineered.pkl'))
        joblib.dump(df_test, os.path.join(self.path_to_save, r'df_test_' + self.target + '_engineered.pkl'))

        print('Result saved as: df_train_' + self.target + '_engineered.pkl')
        print('Result saved as: df_test_' + self.target + '_engineered.pkl')