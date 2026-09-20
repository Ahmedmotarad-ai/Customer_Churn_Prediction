import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "tasks" / "Telco_Customer_Churn_Dataset.csv"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # also exercises startup/shutdown lifespan
        yield c


@pytest.fixture(scope="module")
def known_churner():
    df = pd.read_csv(CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = (
        df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0))
    row = df.drop(columns=["customerID", "Churn", "gender"]).iloc[0].to_dict()
    return {k: (int(v) if k in ("SeniorCitizen", "tenure") else
                float(v) if k in ("MonthlyCharges", "TotalCharges") else v)
            for k, v in row.items()}


@pytest.fixture(scope="module")
def low_risk():
    return dict(known_churner_base(), tenure=70, InternetService="DSL",
                Contract="Two year",
                PaymentMethod="Credit card (automatic)",
                MonthlyCharges=60.0, TotalCharges=4200.0)


def known_churner_base():
    return dict(SeniorCitizen=0, Partner="Yes", Dependents="No",
                PhoneService="Yes", MultipleLines="No",
                OnlineSecurity="Yes", OnlineBackup="Yes",
                DeviceProtection="Yes", TechSupport="Yes",
                StreamingTV="No", StreamingMovies="No",
                PaperlessBilling="No")
