"""Churn prediction API.

Run:  uvicorn app:app --reload   (from inside the api/ folder)
Docs: http://127.0.0.1:8000/docs
Requires model.pkl next to this file (create it with: python train.py).
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model.pkl"


class CustomerFeatures(BaseModel):
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class ChurnPrediction(BaseModel):
    churn: int
    churn_label: str
    churn_probability: float


app = FastAPI(title="Telco Churn API")


def get_bundle():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="model.pkl not found — run 'python train.py' first.",
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures):
    bundle = get_bundle()
    row = pd.DataFrame([customer.model_dump()])[bundle["feature_order"]]
    proba = float(bundle["pipeline"].predict_proba(row)[0, 1])
    label = int(proba >= 0.5)
    return ChurnPrediction(
        churn=label,
        churn_label="Yes" if label else "No",
        churn_probability=round(proba, 4),
    )
