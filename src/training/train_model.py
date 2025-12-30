import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_full_pipeline(X_train_res, y_train_res, X_val, y_val, X_test, y_test):
    """
    Model eğitimi, Checkpoint yönetimi, MLflow loglama ve Yerel Kayıt.
    """
    with mlflow.start_run(run_name="Model_Training_Subrun", nested=True):
        mlflow.log_params({
            "rf_n_estimators": 100,
            "xgb_learning_rate": 0.1,
            "ensemble_type": "Soft Voting"
        })

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_res, y_train_res)
        print("Random Forest (Bagging) model trained successfully.")

        checkpoint_path = "xgb_checkpoint.json"

        xgb_initial = XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
        xgb_initial.fit(X_train_res, y_train_res)
        xgb_initial.save_model(checkpoint_path)
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        xgb_final = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_final.fit(X_train_res, y_train_res, xgb_model=checkpoint_path)
        print("XGBoost (Boosting) model trained by resuming from checkpoint.")

        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb_final)],
            voting='soft'
        )
        ensemble.fit(X_train_res, y_train_res)
        print("Ensemble (Voting) model created and trained successfully.")

        preds = ensemble.predict(X_test)
        probs = ensemble.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "auc_roc": roc_auc_score(y_test, probs)
        }

        for name, val in metrics.items():
            mlflow.log_metric(name, val)
            print(f"{name.upper()}: {val:.4f}")

        mlflow.sklearn.log_model(
            ensemble,
            "final_model",
            registered_model_name="AdClickPredictionModel"
        )

        model_save_path = os.path.join(BASE_DIR, "final_deployment_model.pkl")
        joblib.dump(ensemble, model_save_path)
        print(f"Model successfully saved to local path: {model_save_path}")

        return rf, xgb_final, ensemble
