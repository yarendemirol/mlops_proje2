from sklearn.metrics import accuracy_score


def run_quality_check(y_true, y_pred, threshold=0.85):
    """
    Döküman III.3: Continued Model Evaluation & Fallback
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"Current Model Accuracy: {acc:.4f}")

    if acc < threshold:
        print("[FALLBACK] Performance below threshold! Rolling back to Baseline Model.")
        return "TRIGGER_FALLBACK"

    print("[SUCCESS] Model health is good.")
    return "HEALTHY"