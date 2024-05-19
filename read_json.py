"""
Read json files

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  

import os
import json
import pandas as pd
import joblib
import re


class read_json_files():
    
    def __init__(self, path_to_json, path_to_save):
        
        self.path_to_json = path_to_json
        self.path_to_save = path_to_save

        self.json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

        print('Path to json: ', self.path_to_json)
        print('Save path: ', self.path_to_save)
        print('No. of json:', len(self.json_files))


    def load_files(self):
        # Get Json into pandas dataframe
        for i in range(len(self.json_files)):
            globals()[f'df_{i}'] = pd.read_json(self.path_to_json + self.json_files[i], lines=True)
            
            if i == 0:
                self.df_merged = globals()[f'df_{i}']
            
            else:
                self.df_merged = pd.concat(
                                            [self.df_merged, globals()[f'df_{i}']], 
                                            axis = 0,
                                            join = 'inner',
                                            ignore_index=True
                                            )
        
        # Chemical formula
        formula = []

        for s in self.json_files:
            
            f = re.sub(r"_.*", "", s)
            
            formula.append(f)
        
        # Deuterium
        forbidden = ['D' + str(i) for i in range(100)]
        reformatted_formula = []

        for i in formula:
            if any(Deuterium in i for Deuterium in forbidden):
                i = i.replace("D", "H")
            
            reformatted_formula.append(i)
            
            
        self.df_formula = pd.DataFrame({'formula':reformatted_formula})
        
        self.df_formula = self.df_formula.merge(self.df_merged, left_index=True, right_index=True)
        
        print('df shape:', self.df_formula.shape)
            
        
        return self.df_formula


    def save(self):
        joblib.dump(self.df_formula, self.path_to_save + 'dft_band_gap_SangtaeKim.pkl')
        
        print('Files saved:', 'dft_band_gap_SangtaeKim.pkl')




