"""
Module to create Pymatgen material objects

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""

import os                                                                                                                                                                             
import pandas as pd
from pymatgen import core
import requests
import joblib
import pathlib

from pymatgen.core import Composition, Element    

from matminer.featurizers.conversions import StructureToComposition, StructureToOxidStructure, CompositionToOxidComposition, CompositionToStructureFromMP


class material_object():
    """
    Class to generate Pymatgen materials object   

    args: 
        (1) df (pandas.Dataframe) - dataframe with chemical information 

    return: 
        (1) pandas.Dataframe of features (pkl and/or csv)
    """


    def __init__(self, df):
        self.df = df
        self.df = self.df.fillna(0)
        self.cur_dir = pathlib.Path().resolve()



    def movecol(self, cols_to_move = [], ref_col = '', place = 'after'):
        """
        Function to rearrange columns

        arg: 
            (a) cols_to_move (list) - list of columns to move
            (b) ref_col (type:str) - reference column 
            (c) place (type:str) - whether to move the specified columns 'before' or 'after' the reference column (set to 'after' as default)

        return:
            (a) pandas.Dataframe
        """

        cols = self.df.columns.tolist()

        if place == 'after':
            s1 = cols[:list(cols).index(ref_col) + 1]
            s2 = cols_to_move


        if place == 'before':
            s1 = cols[:list(cols).index(ref_col)]
            s2 = cols_to_move + [ref_col]
        

        s1 = [i for i in s1 if i not in s2]
        s3 = [i for i in cols if i not in s1 + s2]
        

        return self.df[s1 + s2 + s3]

    

    def formula2composition(self, formula_col='formula', composition_col='composition'):
        '''
        Convert chemical formula to Composition Pymatgen object

        arg: 
            (a) formula_col (str) - Column name that contains the formula

        return:
            (a) pandas.Dataframe
        '''
        # Create new column
        self.df[composition_col] = self.df[formula_col] 

        # Create composition object 
        self.df[composition_col] = self.df[composition_col].apply(lambda x: core.Composition(x))

        return self.df



    def structure2composition(self, structure_col='structure'):
        '''
        Convert structure object to Composition Pymatgen object

        arg: 
            (a) structure_col (str) - Column name that contains the structure

        return:
            (a) pandas.Dataframe
        '''

        cf = StructureToComposition(target_col_id='composition')

        # Create composition object 
        self.df = cf.featurize_dataframe(self.df, structure_col, ignore_errors=True)

        return self.df



    def structure2oxidstructure(self, structure_col='structure'):
        '''
        Convert structure object to Oxidation Structure Pymatgen object

        arg: 
            (a) structure_col (str) - Column name that contains the structure

        return:
            (a) pandas.Dataframe
        '''

        # Create oxidation composition object 
        cf = StructureToOxidStructure(target_col_id='oxidation_structure')
        
        self.df = cf.featurize_dataframe(self.df, structure_col, ignore_errors=True)

        return self.df



    def composition2oxidcomposition(self, composition_col='composition'):
        '''
        Convert Composition object to Oxidation Composition Pymatgen object

        arg: 
            (a) structure_col (str) - Column name that contains the structure

        return:
            (a) pandas.Dataframe
        '''

        # Create oxidation composition object 
        cf = CompositionToOxidComposition(target_col_id='oxidation_composition')
        
        self.df = cf.featurize_dataframe(self.df, composition_col, ignore_errors=True)

        return self.df



    def composition2structure_fromMP(self, composition_col='composition', mapi_key='lZCh9ke4qRxMQO16'):
        '''
        Convert Composition object to Structure object from Materials Project

        arg: 
            (a) structure_col (str) - Column name that contains the structure

        return:
            (a) pandas.Dataframe
        '''

        # Create oxidation composition object 
        cf = CompositionToStructureFromMP(target_col_id='structure', mapi_key=mapi_key)
    
        self.df = cf.featurize_dataframe(self.df, composition_col, ignore_errors=True)

        return self.df


    
    def save(self, name, csv=False):

        #Save data as csv
        joblib.dump(self.df, os.path.join(self.cur_dir, str(name) + '.pkl'))

        print('Data saved as:', str(name) + '.pkl')

        if csv == True:
            self.df.to_csv(os.path.join(self.cur_dir, str(name) +  + '.csv'))

        print('Data saved as:', str(name) + '.csv')

