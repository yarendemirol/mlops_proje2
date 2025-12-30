import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from prefect import flow, task
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


from src.features.build_features import apply_feature_engineering
from src.training.train_model import train_full_pipeline
from src.monitoring.quality_check import run_quality_check


@task(name="Data_Ingestion_and_Processing")
def data_step(file_name):
    """Veriyi okur ve Hashing Trick uygular [Döküman III.1]"""
    path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    processed_df = apply_feature_engineering(df)
    return processed_df


@task(name="Model_Training_Registry_and_Metrics")
def training_step(df):
    """Eğitim, Metrik Loglama ve Registry [Döküman III.2 & II.1]"""
    from sklearn.model_selection import train_test_split

    X = df.drop('Clicked on Ad', axis=1)
    y = df['Clicked on Ad']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    rf, xgb, ensemble = train_full_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    preds = ensemble.predict(X_test)
    probs = ensemble.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "auc_roc": roc_auc_score(y_test, probs)
    }

    mlflow.log_metrics(metrics)
    print(f"Metrics successfully logged to MLflow: {metrics}")

    mlflow.sklearn.log_model(
        ensemble,
        "final_model",
        registered_model_name="AdClickPredictionModel"
    )

    return ensemble, X_test, y_test


@flow(name="MLOps_Level2_Final_Execution")
def main_training_flow():
    """Ana Orchestration Flow"""

    mlruns_path = os.path.join(BASE_DIR, "mlruns")
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment("Ad_Click_Production_Project")

    with mlflow.start_run(run_name="Full_Execution_With_Metrics"):
        processed_df = data_step("processed_adv_data.csv")

        model, X_test, y_test = training_step(processed_df)

        y_pred = model.predict(X_test)
        status = run_quality_check(y_test, y_pred)

        mlflow.log_param("final_status", status)
        mlflow.set_tag("owner", "yarendemirol")
        mlflow.set_tag("mlops_level", "2")

        mlflow.end_run()
        print(f"Pipeline finished successfully. Final status: {status}")


if __name__ == "__main__":
    main_training_flow()
