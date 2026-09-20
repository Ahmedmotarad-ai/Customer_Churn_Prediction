"""Train the final churn pipeline and save it to model.pkl.

Reproduces the notebook cleaning (Task 1-3 decisions) and trains the
winning LogisticRegression (C=10 from tuning) on the FULL dataset,
which is the standard choice for a production artifact.
Run:  python train.py   (from inside the api/ folder or anywhere)
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "tasks" / "Telco_Customer_Churn_Dataset.csv"
OUT = BASE / "model.pkl"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def main():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = (
        df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0)
    )
    df = df.drop(columns=["customerID"])
    # Task 3 decision: gender carries no signal (churn gap ~0.76pp)
    X = df.drop(columns=["Churn", "gender"])
    y = (df["Churn"] == "Yes").astype(int)

    categorical = [c for c in X.columns if c not in NUMERIC]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
        ]
    )
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                     C=10.0, random_state=42)),
    ])
    pipe.fit(X, y)
    joblib.dump({"pipeline": pipe, "feature_order": list(X.columns)}, OUT)
    print("trained on:", X.shape, "| saved:", OUT)


if __name__ == "__main__":
    main()
