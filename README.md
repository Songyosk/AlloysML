# Predictive Modeling of High-Entropy Alloys and Amorphous Metallic Alloys Using Machine Learning

All source code and images are associated with the paper:

"Predictive Modeling of High-Entropy Alloys and Amorphous Metallic Alloys Using Machine Learning" 

By S. G. Jung, G. Jung & J. M. Cole



## Introduction

The description of each file is summarized below:


*(i) create_material_objects.py*

Script to create Pymatgen material objects, which is required before featurizers.py

*(ii) featurizers.py*

Script to create features from Pymatgen Material objects (e.g. composition-based & structure-based). 

*(iii) featurize_custom.py*

Script to create various chemical and symmetry-based features using custom functions

*(iv) feature_analyses.py*

Script to perform statistical feature analyses

*(v) feature_engineering.py*

Script to perform feature engineering. By default, a brute-force method is used.

*(vi) GBFS.py*

Script to perform gradient boosted feature selection, generate feature ranking, and carry out recursive feature selection. See "Gradient Boosted and Statistical Feature Selection" in [https://github.com/Songyosk/GBSFS4MPP](https://github.com/Songyosk/GBFS4MPPML).

*(vii) Multicollinearity_reduction.py*

Script to perform multicollinearity reduction, which includes correlation analysis and hierarchical clustering analysis. Correlation and linkage thresholds are defined to elminate features. 

*(viii) permutation_importance.py*

Script to analyze and identify features that are importance when permutation is performed

*(ix) recursive_feature_elimination.py*

Script to perform recursive feature elimination 

*(x) optimization.py*

Script to perform Bayesian optimization, which determines the architecture of the predictive model based on a defined hyperparameter space.  

*(xi) utilities.py and read_json.py*

Scripts containing helper functions 


## Acknowledgements
J.M.C. is grateful for the BASF/Royal Academy of Engineering Research Chair in Data-Driven Molecular Engineering of Functional Materials, which is partly sponsored by the Science and Technology Facilities Council (STFC) via the ISIS Neutron and Muon Source; this Chair also supports a PhD studentship (for S.G.J.). STFC is also thanked for a PhD studentship that is sponsored by its Scientific Computing Department (for G.J.).


## 🔗 Links
[![portfolio](https://img.shields.io/badge/Research_group-000?style=for-the-badge&logo=ko-fi&logoColor=white)](http://www.mole.phy.cam.ac.uk/)


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
