import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from prefect import flow, task
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.features.prepare_dataset import build_features
from src.training.train_model import train_full_pipeline
from src.monitoring.quality_check import run_quality_check
from src.features.rebalancing import analyze_and_rebalance
from src.evaluation.before_after_analysis import run_before_after_comparison


@task(name="1_Data_Ingestion")
def ingestion_step():
    input_path = os.path.join(BASE_DIR, "data", "advertising.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found at: {input_path}")
    return pd.read_csv(input_path)


@task(name="2_Advanced_Feature_Engineering_and_Rebalancing")
def preparation_step(raw_df):
    processed_df = build_features(raw_df)
    X = processed_df.drop('Clicked on Ad', axis=1)
    y = processed_df['Clicked on Ad']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    upsampled_train = analyze_and_rebalance(train_df)
    X_train_res = upsampled_train.drop("Clicked on Ad", axis=1)
    y_train_res = upsampled_train["Clicked on Ad"]

    run_before_after_comparison(X_train, y_train, X_train_res, y_train_res, X_test, y_test)

    return X_train_res, y_train_res, X_val, y_val, X_test, y_test


@task(name="3_Model_Training_and_Comparison")
def training_step(X_train_res, y_train_res, X_val, y_val, X_test, y_test):
    rf, xgb, ensemble = train_full_pipeline(X_train_res, y_train_res, X_val, y_val, X_test, y_test)

    models_dict = {"Bagging_RF": rf, "Boosting_XGB": xgb, "Ensemble_Voting": ensemble}
    results = []

    for name, model in models_dict.items():
        p = model.predict(X_test)
        pr = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, p)
        f1 = f1_score(y_test, p)
        auc = roc_auc_score(y_test, pr)
        pre = precision_score(y_test, p)
        rec = recall_score(y_test, p)

        mlflow.log_metric(f"{name}_accuracy", acc)
        mlflow.log_metric(f"{name}_f1", f1)
        mlflow.log_metric(f"{name}_auc", auc)
        mlflow.log_metric(f"{name}_precision", pre)  # Ekstra metrik logları
        mlflow.log_metric(f"{name}_recall", rec)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": pre,
            "Recall": rec,
            "F1_Score": f1,
            "AUC_ROC": auc
        })

    results_df = pd.DataFrame(results)

    # --- ESKİ KODDAKİ TÜM AYRI GRAFİKLERİN EKLENDİĞİ KISIM ---
    metrics_list = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]
    colors = ["skyblue", "salmon", "lightgreen", "orange", "plum"]

    for i, metric in enumerate(metrics_list):
        plt.figure(figsize=(8, 5))
        plt.bar(results_df["Model"], results_df[metric], color=colors[i], width=0.6)
        plt.title(f"Model Comparison: {metric}")
        plt.ylim(0.7, 1.05)

        # Grafik ismini oluştur ve kaydet
        plot_filename = f"{metric.lower()}_comparison.png"
        plot_path = os.path.join(BASE_DIR, plot_filename)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.show()  # İlk koddaki gibi ekrana basar
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    return ensemble, X_test, y_test


@task(name="4_Model_Registry_and_Quality_Check")
def registry_step(ensemble, X_test, y_test):
    signature = infer_signature(X_test, ensemble.predict(X_test))

    mlflow.sklearn.log_model(
        ensemble,
        "final_model",
        signature=signature,
        registered_model_name="AdClickPredictionModel"
    )

    y_pred = ensemble.predict(X_test)
    status = run_quality_check(y_test, y_pred)

    mlflow.log_param("final_status", status)
    mlflow.log_param("feature_engineering", "Hashing_Scaling_Rebalancing")
    mlflow.set_tag("owner", "yarendemirol")
    mlflow.set_tag("mlops_level", "2")

    return status


@flow(name="MLOps_Level2_Detailed_Execution")
def main_training_flow():
    mlruns_path = os.path.join(BASE_DIR, "mlruns")
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment("Ad_Click_Production_Project")

    with mlflow.start_run(run_name="Full_Detailed_EndToEnd_Execution"):
        raw_df = ingestion_step()
        X_tr, y_tr, X_v, y_v, X_ts, y_ts = preparation_step(raw_df)
        ensemble_model, X_test, y_test = training_step(X_tr, y_tr, X_v, y_v, X_ts, y_ts)
        final_status = registry_step(ensemble_model, X_test, y_test)

        print(f"Pipeline finished successfully. Final status: {final_status}")


if __name__ == "__main__":
    main_training_flow()
