"""Churn prediction API.

Run:  uvicorn app:app --reload   (from inside the api/ folder)
Docs: http://127.0.0.1:8000/docs
Requires model.pkl next to this file (create it with: python train.py).
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE / "model.pkl"))
DEFAULT_THRESHOLD = 0.5

logger = logging.getLogger("churn-api")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def get_threshold():
    try:
        return float(os.environ.get("CHURN_THRESHOLD", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD


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


_bundle = None


def load_bundle():
    """Load the model artifact once and reuse it (also used at startup)."""
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
        logger.info("model loaded from %s (%d features)",
                    MODEL_PATH, len(_bundle["feature_order"]))
    return _bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_bundle()
    except Exception:
        logger.exception("model failed to load from %s", MODEL_PATH)
    logger.info("startup complete")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Telco Churn API",
    description="Predict telecom customer churn (LogisticRegression pipeline).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_errors(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": "internal server error"})


def get_bundle():
    try:
        return load_bundle()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="model not available — run 'python train.py' first.",
        )
    except Exception:
        logger.exception("model failed to load")
        raise HTTPException(status_code=500,
                            detail="model failed to load.")


@app.get("/health")
def health(response: Response):
    info = {"status": "ok", "model_loaded": MODEL_PATH.exists()}
    if info["model_loaded"]:
        try:
            bundle = load_bundle()
            info["model"] = type(
                bundle["pipeline"].named_steps["model"]).__name__
            info["n_features"] = len(bundle["feature_order"])
        except Exception:
            logger.exception("health check could not load model")
            info.update(status="degraded", model_loaded=False)
    else:
        info["status"] = "degraded"
    if info["status"] != "ok":
        # degraded must fail container health probes (Docker/Render)
        # instead of looking healthy while /predict returns 503
        response.status_code = 503
    return info


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures):
    bundle = get_bundle()
    row = pd.DataFrame([customer.model_dump()])[bundle["feature_order"]]
    proba = float(bundle["pipeline"].predict_proba(row)[0, 1])
    threshold = get_threshold()
    label = int(proba >= threshold)
    logger.info("predict churn=%d proba=%.4f", label, proba)
    return ChurnPrediction(
        churn=label,
        churn_label="Yes" if label else "No",
        churn_probability=round(proba, 4),
    )
