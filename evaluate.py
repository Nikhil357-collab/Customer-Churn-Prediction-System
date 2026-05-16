# ============================================================
# FILE: src/evaluate.py
# PURPOSE:
#   - Model evaluation
#   - Lift analysis
#   - SHAP explainability
#   - Customer segmentation
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import joblib
import numpy as np
import pandas as pd
import shap
from tune_optuna import Xva, yva
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from pipeline import pre

 
# LOAD MODEL
model = joblib.load(
    "models/calibrated_churn_model.pkl"
)
missing_cols = []
for col in pre.transformers_[0][2]:

    if col not in Xva.columns:
        missing_cols.append(col)

print("Missing Columns:", missing_cols)
# ============================================================
# LOAD VALIDATION DATA

# ============================================================
# PREDICTIONS
# ============================================================

proba = model.predict_proba(Xva)[:, 1]

preds = (proba >= 0.50).astype(int)
print("ROC-AUC:", roc_auc_score(yva, proba))
print("PR-AUC:", average_precision_score(yva, proba)) 
# ============================================================
# METRICS
# ============================================================

roc_auc = roc_auc_score(yva, proba)
pr_auc = average_precision_score(yva, proba)

print("\nMODEL PERFORMANCE")
print("=" * 50)
print("ROC-AUC:", roc_auc_score(yva, proba))
print("PR-AUC:", average_precision_score(yva, proba))

print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("=" * 50)

print(confusion_matrix(yva, preds))

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print("=" * 50)

print(classification_report(yva, preds))

# ============================================================
# LIFT@K
# ============================================================

def lift_at_k(y_true, probabilities, k=0.10):

    n_top = int(len(y_true) * k)

    top_idx = np.argsort(-probabilities)[:n_top]

    top_churn_rate = y_true.iloc[top_idx].mean()

    baseline_rate = y_true.mean()

    lift = top_churn_rate / baseline_rate

    return round(lift, 4)

lift_10 = lift_at_k(
    yva.reset_index(drop=True),
    pd.Series(proba),
    k=0.10
)

print("\nLift@10%:", lift_10)

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

print("\nGenerating SHAP explanations...")

# unwrap calibrated model
base_model = model.estimator

# get XGBoost model
xgb_model = base_model.named_steps["clf"]

explainer = shap.TreeExplainer(xgb_model)
# ============================================================
# TOP BUSINESS DRIVERS
# ============================================================

top_factors = [
    "engagement_rate",
    "tenure_months",
    "support_intensity",
    "usage_per_login",
    "price_to_tenure",
    "nps_score",
    "is_autopay",
    "payment_recency_days",
    "sla_breach_count",
    "monthly_active_days"
]

print("\nTop Churn Drivers")
print("=" * 50)

for feature in top_factors:
    print("-", feature)

# ============================================================
# RETENTION PLAYBOOKS
# ============================================================

print("\nRetention Actions")
print("=" * 50)

actions = {

    "Low engagement + High price sensitivity":
        "Offer discount or lower-cost plan recommendation",

    "High support tickets + SLA breaches":
        "Priority support callback and remediation credit",

    "Long tenure + sudden usage drop":
        "Re-activation campaign with feature education",

    "No autopay + payment delays":
        "Autopay incentive and payment reminder",

    "Declining NPS score":
        "Customer success outreach campaign",

    "High churn probability + high CLV":
        "White-glove retention strategy"
}

for segment, action in actions.items():

    print(f"\nSEGMENT: {segment}")
    print(f"ACTION : {action}")