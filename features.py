# ============================================

# FEATURE ENGINEERING MOD

# NumPy for mathematical operations
import numpy as np

# Pandas for dataframe manipulation
import pandas as pd

# FEATURE ENGINEERING FUNCTION==========
import pandas as pd

def align_features(df, feature_cols):
    return df.reindex(columns=feature_cols, fill_value=0)
def add_features(df):

    df = df.copy()

    # -----------------------------
    # SAFE DEFAULT COLUMNS
    # -----------------------------
    defaults = {
        "payment_recency_days": 0,
        "active_days": df.get("tenure_months", 0) * 30,
        "monthly_usage_hours": 0,
        "login_count": 1,
        "support_tickets": 0,
        "sla_breaches": 0,
        "email_clicks": 0,
        "email_opens": 1,
        "billing_amount": 0,
        "tenure_months": 1,
        "is_autopay": 0
    }

    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    # -----------------------------
    # FEATURES
    # -----------------------------

    df["engagement_rate"] = (df["active_days"] / 30).clip(0, 1)

    df["usage_per_login"] = df["monthly_usage_hours"] / (df["login_count"] + 1e-3)

    df["support_intensity"] = df["support_tickets"] + 3 * df["sla_breaches"]

    df["email_ctr"] = df["email_clicks"] / (df["email_opens"] + 1e-3)

    df["price_to_tenure"] = df["billing_amount"] / (df["tenure_months"] + 1e-3)

    df["payment_risk"] = df["payment_recency_days"] * (1 - df["is_autopay"])

    df["low_engagement"] = (df["engagement_rate"] < 0.3).astype(int)

    df["high_support"] = (df["support_intensity"] > 5).astype(int)

    df["loyal_customer"] = (df["tenure_months"] > 24).astype(int)

    df["usage_intensity"] = df["monthly_usage_hours"] / (df["tenure_months"] + 1)

    df["inactive_ratio"] = 1 - df["engagement_rate"]

    df["support_per_login"] = df["support_tickets"] / (df["login_count"] + 1)

    df["revenue_per_usage"] = df["billing_amount"] / (df["monthly_usage_hours"] + 1)

    df["churn_pressure"] = (
        df["high_support"] +
        df["low_engagement"] +
        df["payment_risk"]
    )

    jls_extract_var = "payment_recency_days"
    df["payment_risk"] = df[jls_extract_var] * (1 - df["is_autopay"])
    df["support_pressure"] = df["support_tickets"] / (df["tenure_months"] + 1)
    
    return df