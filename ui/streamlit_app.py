"""Telco churn demo UI (Streamlit).

Run from the repo root:  streamlit run ui/streamlit_app.py
Needs api/model.pkl (create it with: python api/train.py).
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE.parent / "api" / "model.pkl"


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("Telco Customer Churn Prediction")
st.write("Fill in the customer profile and press **Predict** "
         "(model: LogisticRegression, same artifact as the API).")

bundle = load_bundle()
pipe = bundle["pipeline"]

with st.form("customer"):
    st.subheader("Customer")
    SeniorCitizen = st.selectbox("Senior citizen", [0, 1])
    Partner = st.selectbox("Partner", ["No", "Yes"], index=1)
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, value=1)
    st.subheader("Services")
    PhoneService = st.selectbox("Phone service", ["No", "Yes"])
    MultipleLines = st.selectbox(
        "Multiple lines", ["No", "No phone service", "Yes"], index=1)
    InternetService = st.selectbox(
        "Internet service", ["DSL", "Fiber optic", "No"])
    no_net = ["No", "No internet service", "Yes"]
    OnlineSecurity = st.selectbox("Online security", no_net)
    OnlineBackup = st.selectbox("Online backup", no_net, index=2)
    DeviceProtection = st.selectbox("Device protection", no_net)
    TechSupport = st.selectbox("Tech support", no_net)
    StreamingTV = st.selectbox("Streaming TV", no_net)
    StreamingMovies = st.selectbox("Streaming movies", no_net)
    st.subheader("Billing")
    Contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless billing", ["No", "Yes"],
                                      index=1)
    PaymentMethod = st.selectbox("Payment method", [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check"], index=2)
    MonthlyCharges = st.number_input("Monthly charges", min_value=0.0,
                                     value=29.85)
    TotalCharges = st.number_input("Total charges", min_value=0.0,
                                   value=29.85)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure,
        "PhoneService": PhoneService, "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }])[bundle["feature_order"]]
    proba = float(pipe.predict_proba(row)[0, 1])
    label = "Yes — will churn" if proba >= 0.5 else "No — will stay"
    st.metric("Prediction", label, f"{proba:.1%} churn probability")
    st.progress(proba)
    if proba >= 0.5:
        st.warning("High risk: consider a retention offer "
                   "(annual contract, support add-ons).")
    else:
        st.success("Low risk customer.")
