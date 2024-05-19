"""
Module to create features from Pymatgen Material objects (e.g. composition-based & structure-based)

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""

import os
from xmlrpc.client import boolean                                                                                                                                                                             
import pandas as pd
import joblib
import pathlib
import numpy as np

from pymatgen.core import Composition   
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.conversions import StructureToComposition

######## Featurizer based on the Composition object
# Element
from matminer.featurizers.composition import ElementFraction, TMetalFraction, Stoichiometry, BandCenter
# Composition
from matminer.featurizers.composition import ElementProperty, Meredig
# Ion 
from matminer.featurizers.composition import OxidationStates, IonProperty, ElectronAffinity, ElectronegativityDiff
# Alloy
from matminer.featurizers.composition.alloy import WenAlloys, Miedema, YangSolidSolution
# Orbital
from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital
# Packing 
from matminer.featurizers.composition.packing import AtomicPackingEfficiency
# Thermo
from matminer.featurizers.composition.thermo import CohesiveEnergy, CohesiveEnergyMP

######## Featurizer based on the Structure object
# Matrix
from matminer.featurizers.structure.matrix import OrbitalFieldMatrix
# Misc
from matminer.featurizers.structure.misc import XRDPowderPattern
# Order
from matminer.featurizers.structure.order import DensityFeatures, ChemicalOrdering, MaximumPackingEfficiency, StructuralComplexity
# RDF
from matminer.featurizers.structure.rdf import ElectronicRadialDistributionFunction
# Site
from matminer.featurizers.structure.sites import SiteStatsFingerprint
# Symmetry
from matminer.featurizers.structure.symmetry import GlobalSymmetryFeatures, Dimensionality
# Composition
from matminer.featurizers.structure.composite import JarvisCFID


class prepare_to_featurize():
    """
    Class to generate features  

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



    def OHE(self, *args, **kwargs):
        """
        One-hot-encoding of categorical columns

        return:
            (a) pandas.Dataframe
        """
        target = kwargs.get('target')
        print(target)
        
        categorical_cols = [col for col, dt in self.df.dtypes.items() 
                            if dt == object 
                            and col != 'formula' 
                            and col != 'structure'
                            and col != 'original_formula'
                            and col != target
                            ]
        
        # num_cols = self.df._get_numeric_data().columns
    
        # categorical_cols = list(set(self.df.columns) - set(num_cols))
        
        # if col is not None:
        #     categorical_cols.remove(col)

        print('No. of categorical features:', len(categorical_cols))

        self.df = pd.get_dummies(data = self.df, columns = categorical_cols, prefix_sep = '_ohe_', drop_first = False)

        self.ohe_cols = [i for i in self.df.columns if '_ohe_' in i]


        return self.df, self.ohe_cols



    def drop(self, cols=[]):
        '''
        Function to drop unwanted columns

        return:
            (a) pandas.Dataframe
        '''

        self.df = self.df.drop(cols, axis=1)

        return self.df
    


    def generate_oxid_composition_features(self, composition_col='oxidation_composition'):
        '''
        Generate features using the Composition object. Additional featurizers by uncommenting those excluded.

        arg: 
            (a) oxidation_features (boolean) - generate ion featuresgenerate_composition_features

        return:
            (a) pandas.Dataframe
        '''

        oxid_composition_featurizer = MultipleFeaturizer([
                                                            OxidationStates(), # Ion 
                                                            IonProperty(), # Ion 
                                                            ElectronAffinity(), # Ion 
                                                            ElectronegativityDiff(), # Ion 
                                                        ])

        self.df = oxid_composition_featurizer.featurize_dataframe(self.df, composition_col, ignore_errors=True)

        feature_len = len(oxid_composition_featurizer.feature_labels())

        print('Total no. of features generated:', feature_len)


        # Inf to NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN to zero
        self.df = self.df.fillna(0)

        return self.df



    def generate_oxid_composition_features_with_composition(self, composition_col='composition'):
        '''
        Generate features using the Composition object. Additional featurizers by uncommenting those excluded.

        arg: 
            (a) oxidation_features (boolean) - generate ion features

        return:
            (a) pandas.Dataframe
        '''

        oxid_composition_featurizer = MultipleFeaturizer([
                                                            OxidationStates(), # Ion 
                                                            IonProperty(), # Ion 
                                                            ElectronAffinity(), # Ion 
                                                            ElectronegativityDiff(), # Ion 
                                                        ])

        self.df = oxid_composition_featurizer.featurize_dataframe(self.df, composition_col, ignore_errors=True)

        feature_len = len(oxid_composition_featurizer.feature_labels())

        print('Total no. of features generated:', feature_len)


        # Inf to NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN to zero
        self.df = self.df.fillna(0)

        return self.df


    def generate_composition_features(self, composition_col = 'composition'):
        '''
        Generate features using the Composition object. Additional featurizers by uncommenting those excluded.

        arg: 
            (a) oxidation_features (boolean) - generate ion features

        return:
            (a) pandas.Dataframe
        '''

        composition_featurizer = MultipleFeaturizer([
                                                        ElementFraction(), # Element
                                                        TMetalFraction(), # Element
                                                        Stoichiometry(), # Element
                                                        BandCenter(), # Element
                                                        ElementProperty.from_preset('magpie'), # Composition
                                                        ElementProperty.from_preset('matminer'), # Composition
                                                        ElementProperty.from_preset('deml'), # Composition
                                                        ElementProperty.from_preset('megnet_el'), # Composition
                                                        Meredig(), # Composition
                                                        YangSolidSolution(), # Alloy
                                                        AtomicOrbitals(), # Orbital
                                                        ValenceOrbital(), # Orbital
                                                        AtomicPackingEfficiency(), # Packing 
                                                        
                                                        # ### Additional featurizers:
                                                        WenAlloys(), # Alloy 
                                                        Miedema(), # Alloy
                                                        CohesiveEnergy(mapi_key='lZCh9ke4qRxMQO16'), # Thermo
                                                        CohesiveEnergyMP(mapi_key='lZCh9ke4qRxMQO16') # Thermo
                                                    ])

        self.df = composition_featurizer.featurize_dataframe(self.df, composition_col, ignore_errors=True)

        feature_len = len(composition_featurizer.feature_labels())

        print('Total no. of features generated:', feature_len)


        # Inf to NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN to zero
        self.df = self.df.fillna(0)

        return self.df, composition_featurizer.feature_labels()



    def generate_structural_features(self):
        '''
        Generate features using the Composition object. See the list below.

        arg: 
            (a) oxidation_features (boolean) - generate ion features

        return:
            (a) pandas.Dataframe
        '''

        structural_featurizer = MultipleFeaturizer([
                                                        OrbitalFieldMatrix(), # Matrix
                                                        XRDPowderPattern(), # Misc
                                                        DensityFeatures(), # Order
                                                        ChemicalOrdering(), # Order
                                                        MaximumPackingEfficiency(), # Order
                                                        StructuralComplexity(), # Order
                                                        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"), # Site
                                                        SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"), # Site
                                                        GlobalSymmetryFeatures(), # Symmetry
                                                        Dimensionality() # Symmetry
                                                    ])

        self.df = structural_featurizer.featurize_dataframe(self.df, 'structure', ignore_errors=True)

        feature_len = len(structural_featurizer.feature_labels())

        print('Total no. of features generated:', feature_len)


        # Inf to NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN to zero
        self.df = self.df.fillna(0)

        return self.df




    def generate_oxid_structural_features(self):
        '''
        Generate Electronic Radial Distribution Function features using the Composition object. 

        arg: 
            (a) oxidation_features (boolean) - generate ion features

        return:
            (a) pandas.Dataframe
        '''

        structural_featurizer = MultipleFeaturizer([
                                                    ElectronicRadialDistributionFunction(), # RDF
                                                    ]) 

        self.df = structural_featurizer.featurize_dataframe(self.df, 'oxidation_structure', ignore_errors=True)

        feature_len = len(structural_featurizer.feature_labels())

        print('Total no. of features generated:', feature_len)


        # Inf to NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN to zero
        self.df = self.df.fillna(0)

        return self.df



    def generate_JarvisCFID_features(self):
        '''
        Generate Jarvis CFID features using the Composition object. 

        arg: 
            (a) oxidation_features (boolean) - generate ion features

        return:
            (a) pandas.Dataframe
        '''

        structural_featurizer = MultipleFeaturizer([
                                                    JarvisCFID(), # Composition
                                                    ]) 

        self.df = structural_featurizer.featurize_dataframe(self.df, 'structure', ignore_errors=True)

        feature_len = len(structural_featurizer.feature_labels())

        print('Total no. of features generated:', feature_len)


        # Inf to NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN to zero
        self.df = self.df.fillna(0)

        return self.df



    def custom_features(self, composition_col_exist, composition_col='composition'):
        '''
        Create custom features which includes: 
        (a) weight, 
        (b) total electrons,
        (c) electronegativity, 
        (d) noble_gas, 
        (e) transition_metal, 
        (f) post_transition_metal, 
        (g) rare_earth_metal, 
        (h) metal, metalloid,  
        (i) alkali, 
        (j) alkaline, 
        (k) halogen,
        (l) chalcogen, 
        (m) lanthanoid, 
        (n) actinoid, 
        (o) quadrupolar, 
        (p) s-block, 
        (q) p-block, 
        (r) d-block, 
        (s) f-block, 
        '''

        # Check if Composition contains any elements matching a given category

        category = [
            'noble_gas', 'transition_metal', 'post_transition_metal', 'rare_earth_metal', 'metal', 'metalloid', \
            'alkali', 'alkaline', 'halogen', 'chalcogen', 'lanthanoid', 'actinoid', 'quadrupolar', 's-block', 'p-block', \
            'd-block', 'f-block'
            ]


        if composition_col_exist is False:
            
            cf = StructureToComposition(target_col_id='composition')

            self.df = cf.featurize_dataframe(self.df, 'structure', ignore_errors=True)


        # Generate total molecular weight of Composition
        self.df['weight'] = self.df[composition_col]
        self.df['weight'] = self.df['weight'].map(lambda x: x.weight)


        # Generate total electrons
        self.df['total_e'] = self.df[composition_col]
        self.df['total_e'] = self.df['total_e'].map(lambda x: x.total_electrons)


        # Generate average electronegativity of the composition
        self.df['avg_electroneg'] = self.df[composition_col]
        self.df['avg_electroneg'] = self.df['avg_electroneg'].map(lambda x: x.average_electroneg)

        for c in category:
            # print(c)
            self.df[c] = self.df[composition_col]
            self.df[c] = self.df[c].map(lambda x: x.contains_element_type(c))
            self.df[c] = self.df[c].astype(int)

        return self.df

    
    def save(self, name, csv=False):

        #Save data as csv
        joblib.dump(self.df, os.path.join(self.cur_dir, str(name) + '.pkl'))

        print('Data saved as:', str(name) + '.pkl')

        if csv == True:
            self.df.to_csv(os.path.join(self.cur_dir, str(name) +  + '.csv'))

        print('Data saved as:', str(name) + '.csv')


        

