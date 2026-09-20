"""Model sanity tests: artifact integrity + domain behavior."""
import joblib
import pandas as pd

from app import MODEL_PATH
from conftest import CSV


def _bundle():
    return joblib.load(MODEL_PATH)


def test_artifact_structure():
    b = _bundle()
    assert list(b["pipeline"].named_steps) == ["preprocessor", "model"]
    assert len(b["feature_order"]) == 18
    assert "gender" not in b["feature_order"]


def test_probabilities_bounded_on_real_rows():
    b = _bundle()
    df = pd.read_csv(CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = (
        df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0))
    X = df.drop(columns=["customerID", "Churn", "gender"])[b["feature_order"]]
    proba = b["pipeline"].predict_proba(X.sample(50, random_state=42))
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    assert abs(proba.sum(axis=1) - 1.0).max() < 1e-6


def test_high_risk_scores_above_low_risk():
    b = _bundle()
    base = dict(SeniorCitizen=0, Partner="Yes", Dependents="No",
                PhoneService="Yes", MultipleLines="No", OnlineSecurity="Yes",
                OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes",
                StreamingTV="No", StreamingMovies="No", PaperlessBilling="No")
    high = dict(base, tenure=1, InternetService="Fiber optic",
                Contract="Month-to-month", PaymentMethod="Electronic check",
                MonthlyCharges=90.0, TotalCharges=90.0)
    low = dict(base, tenure=70, InternetService="DSL", Contract="Two year",
               PaymentMethod="Credit card (automatic)",
               MonthlyCharges=60.0, TotalCharges=4200.0)
    p_high = float(b["pipeline"].predict_proba(
        pd.DataFrame([high])[b["feature_order"]])[0, 1])
    p_low = float(b["pipeline"].predict_proba(
        pd.DataFrame([low])[b["feature_order"]])[0, 1])
    assert p_high > p_low
