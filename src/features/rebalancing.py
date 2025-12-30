import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample


def analyze_and_rebalance(df):
    """Döküman III.2: Data Imbalance & Rebalancing Pattern [cite: 38]"""

    class_dist_df = df['Clicked on Ad'].value_counts().reset_index()
    class_dist_df.columns = ['Class', 'Count']
    print("\n[Kişi 3] Class Distribution (Original)")
    print(class_dist_df.to_string(index=False))


    plt.figure(figsize=(6, 4))
    plt.bar(class_dist_df['Class'], class_dist_df['Count'])
    plt.title("Class Distribution Before Rebalancing")
    plt.show()


    majority = df[df['Clicked on Ad'] == 0]
    minority = df[df['Clicked on Ad'] == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

    return pd.concat([majority, minority_upsampled])