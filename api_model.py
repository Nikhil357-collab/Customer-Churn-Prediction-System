import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from features import add_features

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/calibrated_churn_model.pkl")

app = FastAPI(title="Churn Dashboard API")

# ============================================================
# CORS (IMPORTANT FOR NEXT.JS)
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# INPUT SCHEMA
# ============================================================

class CustomerInput(BaseModel):
    engagement_rate: float
    tenure_months: float
    support_intensity: float
    usage_per_login: float
    price_to_tenure: float
    nps_score: float
    is_autopay: int
    payment_recency_days: float

# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/predict")
def predict(customer: CustomerInput):

    df = pd.DataFrame([customer.model_dump()])
    df = add_features(df).fillna(0)

    proba = model.predict_proba(df)[:, 1][0]

    return {
        "churn_probability": float(proba)
    }

# ============================================================
# TOP CHURNERS (FOR DASHBOARD TABLE)
# ============================================================

@app.post("/top-churners")
def top_churners(customers: List[CustomerInput]):

    df = pd.DataFrame([c.model_dump() for c in customers])
    df = add_features(df).fillna(0)

    probs = model.predict_proba(df)[:, 1]

    df["churn_probability"] = probs

    df = df.sort_values("churn_probability", ascending=False)

    return df.to_dict(orient="records")

# ============================================================
# LIVE DASHBOARD DATA (SIMULATION ENDPOINT)
# ============================================================

@app.get("/live-metrics")
def live_metrics():

    import random

    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "avg_churn": random.uniform(0.2, 0.6),
        "high_risk_customers": random.randint(10, 50),
        "total_customers": 1000
    }