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

def test_missing_values_handling():
    """Veri setinde eksik (NaN) değerler olduğunda sistemin çökmediğini test eder."""
    test_df = pd.DataFrame({
        'Age': [np.nan, 25],
        'City': [np.nan, 'Ankara'],
        'Country': ['Turkey', np.nan],
        'Ad Topic Line': ['Ad1', 'Ad2']
    })
    
    processed = apply_feature_engineering(test_df)
    
    assert 'Age_Bucket' in processed.columns
    assert processed['Age_Bucket'].isnull().sum() == 0  
def test_feature_cross_logic():
    """Feature Cross ve Hashing birleşiminin çıktı üretip üretmediğini test eder."""
    test_df = pd.DataFrame({
        'Age': [30],
        'Ad Topic Line': ['Special Offer'],
        'Country': ['Germany']
    })
    
    processed = apply_feature_engineering(test_df)
    
    hash_cols = [c for c in processed.columns if 'hash' in c]
    assert len(hash_cols) > 0
