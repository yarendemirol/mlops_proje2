import pandas as pd
import numpy as np
from src.features.build_features import apply_feature_engineering


def test_age_bucketing():

    test_df = pd.DataFrame({'Age': [20, 35, 60]})
    processed = apply_feature_engineering(test_df)


    assert 'Age_Bucket' in processed.columns

    assert processed['Age_Bucket'].isnull().sum() == 0


def test_hashing_output():
    test_df = pd.DataFrame({
        'Age': [30],
        'City': ['Istanbul'],
        'Country': ['Turkey'],
        'Ad Topic Line': ['New Course']
    })
    processed = apply_feature_engineering(test_df)

    assert 'City_hash_0' in processed.columns
    assert 'City' not in processed.columns