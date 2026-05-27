
"""
HOW TO USE IN NOTEBOOK:
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from ml_utils import clean_data, quick_eda, profile_data
"""

import pandas as pd
from ydata_profiling import ProfileReport

def profile_data(data, title):
    profile = ProfileReport(data, title=title)
    return profile

def quick_eda(data):
    print("=" * 40)
    
    shape = data.shape
    print("Shape:", shape)
    print("=" * 40)

    dup = data.duplicated().sum()
    print("Duplicates:", dup)
    print("=" * 40)

    miss_val = data.isna().sum()
    print("Missing Values:\n", miss_val)
    print("=" * 40)

    prcnt_miss_val = (data.isna().sum() / len(data) * 100).round(2)
    print("Missing Values %:\n", prcnt_miss_val)
    print("=" * 40)

    dtyp = data.dtypes
    print("Data Types:\n", dtyp)
    print("=" * 40)

    uni_val = data.nunique()
    print("Unique Values:\n", uni_val)
    print("=" * 40)

def clean_data(data, threshold):    
    cols_to_drop = data.columns[data.isna().sum() / len(data) > threshold]
    data = data.drop(columns=cols_to_drop)  
    print(f"Missng Values was higher than {threshold}, so they have been dropped.")
    
    cols_to_fill = data.columns[data.isna().sum() / len(data) < threshold]
    data[cols_to_fill] = data[cols_to_fill].fillna(data[cols_to_fill].median())

    dups = data.duplicated().sum()
    data = data.drop_duplicates()
    print(f"{dups} have been dropped.")

    return data

# FUNCTIONS to do:
# encode_features, scale_features, detect_outliers

