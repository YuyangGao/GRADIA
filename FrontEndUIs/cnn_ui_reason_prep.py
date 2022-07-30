"""
CNN UI Reason Prep for Initialization
"""

import numpy as np
import pandas as pd
import random

## Load image filenames
idx_list = pd.DataFrame(["demo_img_1.jpg", "demo_img_2.jpg", "demo_img_3.jpg", "demo_img_4.jpg", "demo_img_5.jpg"])
df = idx_list.reset_index(drop=True)

def add_col(df):
    df.columns = ['img_idx']
    for i in range(1, 5):  # 4 selections
        df['a_%s' % i] = np.nan
    
    for i in range(1, 5):
        df['b_%s' % i] = np.nan
        
    for i in range(1, 5):
        df['c_%s' % i] = np.nan

    return df

def make_current(df):
    df_current = pd.DataFrame()
    df_current.loc[0, 'current_order'] = 0
    df_current.loc[0, 'current_idx'] = df['img_idx'][0]
    return df_current

## Random img order
def add_rnd_order(df):
    for i in range(len(df)):
        random.seed(i)
        df.loc[i, 'rnd_order'] = str([0] + random.sample(range(1, 5), 4))  # 1 to 4
    
    return df


df = add_col(df)
df_current = make_current(df)
df_rnd = add_rnd_order(df)

## Save files for UI
df_current.to_csv("output/current.csv")
df_rnd.to_csv("output/results.csv")

## Save initial backup files for starting over
df_current.to_csv("output/init_current.csv")
df_rnd.to_csv("output/init_results.csv")



