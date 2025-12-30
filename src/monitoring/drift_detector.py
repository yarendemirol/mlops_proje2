import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def check_model_degradation(y_true, y_pred, threshold=0.90):
    """
    Model performansını kontrol eder. Eğer doğruluk threshold'un altına
    düşerse Algorithmic Fallback tetiklenir.
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"Current Model Accuracy: {acc}")

    if acc < threshold:
        print("!!! WARNING: Performance Degradation Detected !!!")
        print("Action: Triggering Algorithmic Fallback to older version.")
        return True  # Drift var
    return False  # Her şey yolunda
