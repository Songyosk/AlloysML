"""
Utilities

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""


import pandas as pd
import matplotlib.pyplot as plt


def movecol(df, cols_to_move=[], ref_col='', place='After'):
    
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])


def predict(dataframe, model, features, target, new_col_name):

    # Make prediction
    predicted_target = model.predict(dataframe[features])
    print('Target:', target)
    print('No of input:', len(dataframe))
    print('No of prediction:', len(predicted_target))

    # Concat new column to dataframe
    df_temp = pd.DataFrame(data={str(new_col_name): predicted_target}, index=dataframe.index)

    # Concat prediction
    dataframe = pd.concat([dataframe, df_temp], join='outer', axis=1)

    # Move column 
    dataframe = movecol(dataframe, cols_to_move=[str(new_col_name)], ref_col=str(target), place='After')

    return dataframe


def generate_missing_features(df, missing_features):
    
    for f in missing_features:
        sub_features = f.split('/')
        
        df_temp = df[sub_features[0]] / df[sub_features[1]]
        
        df[f] = (df_temp - df_temp.min()) / (df_temp.max() - df_temp.min())
        
    return df