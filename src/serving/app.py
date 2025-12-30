import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.features.build_features import apply_feature_engineering

app = FastAPI(title="Ad Click Prediction - MLOps Level 2")


MODEL_PATH = "final_deployment_model.pkl"
model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    Daily_Time_Spent_on_Site: float
    Age: float
    Area_Income: float
    Daily_Internet_Usage: float
    Male: int
    City: str
    Country: str
    Ad_Topic_Line: str


@app.post("/predict")
def predict(request: PredictionRequest):
    raw_data = pd.DataFrame([request.model_dump()])


    processed_data = apply_feature_engineering(raw_data)


    expected_features = list(model.feature_names_in_)
    final_input = processed_data.reindex(columns=expected_features, fill_value=0)


    prediction = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0].tolist()

    return {
        "prediction": int(prediction),
        "label": "Clicked" if prediction == 1 else "Not Clicked",
        "probability": {"not_clicked": prob[0], "clicked": prob[1]}
    }