# ============================================
# XGBOOST + OPTUNA + CALIBRATION (CLEAN)
# ============================================

import warnings


import joblib
import optuna
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.metrics import recall_score
from features import add_features
from pipeline import pre
warnings.filterwarnings("ignore")

# ============================================
# LOAD DATA
# ============================================

df = pd.read_parquet("data/churn_frame.parquet")
df = add_features(df)

X = df.drop(columns=[
    "churned_next_cycle",
    "cycle_start",
    "cycle_end",
    "customer_id"
])

y = df["churned_next_cycle"]

# ============================================
# TRAIN / VALIDATION SPLIT
# ============================================

Xtr, Xva, ytr, yva = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
# ============================================
# CLASS IMBALANCE
# ============================================

scale_pos_weight = (ytr.value_counts()[0] / ytr.value_counts()[1]) * 2.0
# ============================================
# OPTUNA OBJECTIVE
# ============================================

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 900),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss"
    }

    model = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(**params))
    ])

    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xva)[:, 1]
    preds = (proba >= 0.215).astype(int)

    return recall_score(yva, preds)
# ============================================

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

best_params = study.best_params

def align_features(df, feature_cols):
    return df.reindex(columns=feature_cols, fill_value=0)
# ============================================
# FINAL MODEL
# ============================================

final_model = Pipeline([
    ("pre", pre),
    ("clf", XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight * 1.2,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    ))
])

final_model.fit(Xtr, ytr)

# ============================================
# CALIBRATION
# ============================================

calibrated_model = CalibratedClassifierCV(
    final_model,
    method="isotonic",
    cv=3
)

calibrated_model.fit(Xtr, ytr)

# ============================================
# EVALUATION
# ============================================

proba = calibrated_model.predict_proba(Xva)[:, 1]

pr_auc = average_precision_score(yva, proba)
roc_auc = roc_auc_score(yva, proba)

print("\nFINAL RESULTS")
print("=" * 50)
print(f"PR-AUC  : {pr_auc:.4f}")
print(f"ROC-AUC : {roc_auc:.4f}")

# ============================================
def find_best_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    idx = f1.argmax()
    return thresholds[max(idx - 1, 0)]

best_threshold = find_best_threshold(yva, proba)
print("\nBest Threshold:", best_threshold)

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(calibrated_model, "models/calibrated_churn_model.pkl")

print("\nModel saved successfully.")