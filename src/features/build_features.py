import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher


HASH_CONFIG = {
    "Ad Topic Line": 128,
    "City": 32,
    "Country": 16
}


def apply_feature_engineering(df):
    """
    Hem Eğitim (Training) hem de Tahmin (Inference) sırasında
    aynı işlemleri yapan merkezi fonksiyon.
    """
    df = df.copy()

    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Age_Bucket'] = pd.cut(
            df['Age'],
            bins=[-np.inf, 25, 45, np.inf],
            labels=[0, 1, 2]
        ).astype(int)

    for col, n_feats in HASH_CONFIG.items():
        if col in df.columns:
            hasher = FeatureHasher(n_features=n_feats, input_type="string")
            hashed_data = hasher.transform(
                df[col].astype(str).apply(lambda x: [x])
            )
            hashed_df = pd.DataFrame(
                hashed_data.toarray(),
                columns=[f"{col.replace(' ', '_')}_hash_{i}" for i in range(n_feats)],
                index=df.index
            )
            df = pd.concat([df, hashed_df], axis=1)
            df.drop(columns=[col], inplace=True)

    return df
