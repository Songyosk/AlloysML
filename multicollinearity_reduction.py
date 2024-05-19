"""
Module to reduce multicollinearity within a dataset

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  

import os                                                                                                                                                                                  
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from collections import defaultdict


class multicollinearity_reduction():
    """
    Class to achieve multicollinearity reduction

    args: 
        (1) path_to_file (type:str) - location of the data file with features
        (2) path_to_save (type:str) - location to save new data files
        (3) target (type:str) - name of target variable
        (4) features (list) - list of exploratory features

    return: 
        (1) pandas.Dataframe with collinear features removed
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
        


    def correlation_heatmap(self):
        """
        Generate correlation heat map of the exploratory features
        """

        # List of exploratory features redfined as those with low variance are removed
        self.features = [i for i in self.features if i in self.sample_train.columns]

        # Calculate correlation coeff
        correlations = self.sample_train[self.features].corr()

        # Plot figure
        fig, ax = plt.subplots(figsize = (20,20))

        sns.heatmap(
                    correlations, 
                    vmax = 1.0, 
                    center = 0, 
                    fmt = '.2f', 
                    cmap = "YlGnBu", 
                    square = True, 
                    linewidths = .01, 
                    annot = False, 
                    cbar_kws = {"shrink": .70},
                    xticklabels = True, 
                    yticklabels = True
                    )

        plt.show()

        fig.savefig(os.path.join(self.path_to_save, r'correlation_heatmap.png'), dpi = 300, bbox_inches="tight")

        print('Figure saved as: "correlation_heatmap.png"')



    def correlation_analysis(self, threshold = 0.85):
        """
        Identify features with correlation that is greater than the threshold (default set to 0.85)

        args: 
            (1) threshold (type:float) - correlation threshold to apply

        return: 
            (1) a set of features that are below the correlation threshold
        """

        self.col_corr = set()

        # Compute Pearson's R
        corr_matrix = self.sample_train[self.features].corr() 
        

        # Identify correlated features
        for i in range(len(corr_matrix.columns)): 
            for j in range(i):
                if abs(corr_matrix.iloc[i,j]) > threshold: 
                    colName = corr_matrix.columns[i]
                    self.col_corr.add(colName)

                    # print(corr_matrix.columns[i], ' is correlated with ', corr_matrix.columns[j])

        print('Identified correlated features')


        return self.col_corr



    def apply_correlation_filter(self):
        """
        Remove one of the features when the correlation between a pair of features is greater than the threshold
        """

        # Copy the set of exploratory features
        self.features_v2 = self.features


        # Remove correlated features
        for i in self.col_corr:
            self.features_v2.remove(str(i))

        print('No. of features remaining: ', len(self.features_v2))


        # Save features
        joblib.dump(self.features_v2, os.path.join(self.path_to_save, r'features_selected_from_correlation_analysis_' + self.target + '.pkl'))

        print('Features saved as: features_selected_from_correlation_analysis_' + self.target + '.pkl')


        return self.features_v2



    def hierarchical_cluster_analysis(self,  *args, **kwargs):
        """
        Perform hierarchical cluster analysis & create the corresponding dendrogram 
        """

        x_label_in_numbers = kwargs.get('x_label_in_numbers')
        x_fontsize = kwargs.get('x_fontsize')
        
        # Horizontal line
        horizontal_line = kwargs.get('horizontal_line')
        
        
        # Custom list of x
        x_list = kwargs.get('x_list')
        
        if x_list is not None:
            if isinstance(x_list, list):
                self.x_list = x_list
            else:
                self.x_list = joblib.load(x_list) 
            

        # Create figure
        fig, ax = plt.subplots(figsize = (20, 10))

        if x_fontsize is None:
            fontsize1 = 10
        else:
            fontsize1 = x_fontsize
            
        fontsize2 = 25

        plt.xlabel('Feature number', fontsize = fontsize2)
        plt.ylabel("Ward's linkage distance", fontsize = fontsize2)
        plt.tick_params(direction = "in")
        plt.tick_params(axis='y', direction = "in", labelsize=fontsize2)
        
        if horizontal_line is not None:
            plt.axhline(y=horizontal_line, color='k', linestyle='--')


        # Compute Spearman's R
        self.corr = spearmanr(self.sample_train[self.features_v2]).correlation

    
        # Replace NaN to 0
        self.corr[np.isnan(self.corr)] = 0


        # Ward's linkage distannce based on Spearman's R
        self.corr_linkage = hierarchy.ward(self.corr)


        # Construct corresponding dendrogram
        if x_label_in_numbers == True:
            if x_list is not None:
                hierarchy.dendrogram(
                        self.corr_linkage, 
                        labels = self.x_list, 
                        orientation = 'top', 
                        leaf_rotation = 90, 
                        leaf_font_size = fontsize1
                        )
            else:
                hierarchy.dendrogram(
                    self.corr_linkage, 
                    labels = range(1, len(self.features_v2) + 1), 
                    orientation = 'top', 
                    leaf_rotation = 90, 
                    leaf_font_size = fontsize1
                    )

        else:
            hierarchy.dendrogram(
                                self.corr_linkage, 
                                labels = self.features_v2, 
                                orientation = 'top', 
                                leaf_rotation = 90, 
                                leaf_font_size = fontsize1
                                )


        #final_figure
        fig.savefig(os.path.join(self.path_to_save, r'Dendrogram_' + self.target + '.png'), dpi = 300, bbox_inches="tight")

        print('Figure saved as: Dendrogram_' + self.target + '.png')



    def hierarchical_cluster_map(self):
        """
        Generate hierarchical cluster map
        """

        # Compute Spearman's R
        self.corr = spearmanr(self.sample_train[self.features_v2]).correlation

        # Cluster map
        fig = sns.clustermap(
                            self.corr,
                            method = "ward", 
                            cmap = "YlGnBu", 
                            figsize = (15,15)
                            )

        print('Note: the axex are labelled using the index of the feature columns within the dataset')

        fig.savefig(os.path.join(self.path_to_save, r'hierarchical_cluster_map_' + self.target + '.png'), dpi = 300, bbox_inches="tight")

        print('Figure saved as: hierarchical_cluster_map_' + self.target + '.png')



    def apply_linkage_threshold(self, threshold = 1):
        """
        Apply the linkage threshold and selected features above the threshold

        args: 
            (1) threshold (type:int or float) - linkage threshold to apply for feature selection

        return:
            (1) list of features with correlated features removed
        """

        # Obtain cluster IDs
        cluster_ids = hierarchy.fcluster(
                                        self.corr_linkage, 
                                        t = threshold, 
                                        criterion = 'distance'
                                        )

        cluster_id_to_feature_ids = defaultdict(list)


        # Obtain the index of features
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        
        selected_features = [value[0] for value in cluster_id_to_feature_ids.values()]


        # Define new set of features w
        self.features_v3 = []

        for i in selected_features:
            self.features_v3.append(self.features_v2[i])

            
        print('Number of features remaining: ', len(self.features_v3))
        print('Features saved as features_selected_from_hierarchical_analysis_' + self.target + '_threshold_' + str(threshold) +  '.pkl')

        joblib.dump(self.features_v3, os.path.join(self.path_to_save, r'features_selected_from_hierarchical_analysis_' + self.target + '_threshold_' + str(threshold) +  '.pkl'))

        return self.features_v3
