import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

def run_before_after_comparison(X_train, y_train, X_train_res, y_train_res, X_test, y_test):
    results = []
    sets = [
        ("Original (Imbalanced)", X_train, y_train, "Kisi3_Before_Rebalancing"),
        ("Rebalanced (Upsampling)", X_train_res, y_train_res, "Kisi3_After_Rebalancing")
    ]

    for label, x_tr, y_tr, run_name in sets:
        with mlflow.start_run(run_name=run_name, nested=True):
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(x_tr, y_tr)

            p = model.predict(X_test)
            pr = model.predict_proba(X_test)[:, 1]

            res = {
                "Dataset": label,
                "Accuracy": accuracy_score(y_test, p),
                "Precision": precision_score(y_test, p),
                "Recall": recall_score(y_test, p),
                "F1 Score": f1_score(y_test, p),
                "AUC-ROC": roc_auc_score(y_test, pr)
            }
            results.append(res)

            mlflow.log_params(RF_PARAMS)
            for k, v in res.items():
                if k != "Dataset": mlflow.log_metric(k.lower().replace(" ", "_"), v)

    comparison_df = pd.DataFrame(results)
    print("\n" + "=" * 60 + "\nKİŞİ 3: BEFORE vs AFTER COMPARISON (MLflow Nested Runs)\n" + "=" * 60)
    print(comparison_df.round(4).to_string(index=False))

    comparison_df.set_index("Dataset").plot(kind="bar", figsize=(10, 6))
    plt.title("Effect of Rebalancing (Same Model Params)")
    plt.show()