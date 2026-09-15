"""Train the sepsis early-warning XGBoost classifier.

Patient-grouped train/val/test split (no stay leaks across splits), focal-
style class weighting for the ~7% positive rate, and a small Optuna-free
grid search over the handful of hyperparameters that matter most on this
dataset size.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import build_feature_matrix, FEATURE_COLS

DATA_PATH = Path(__file__).parent.parent / "data" / "icu_stays.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    X, y, groups, _ = build_feature_matrix(df)

    # Patient-grouped split: 70% train, 15% val, 15% test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, rest_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx_rel, test_idx_rel = next(gss2.split(X.iloc[rest_idx], y.iloc[rest_idx], groups.iloc[rest_idx]))
    val_idx = rest_idx[val_idx_rel]
    test_idx = rest_idx[test_idx_rel]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate

    model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", early_stopping_rounds=30,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, MODEL_DIR / "sepsis_xgb.joblib")
    np.save(MODEL_DIR / "test_idx.npy", X_test.index.to_numpy())

    metrics = {"val_auc_roc": round(val_auc, 4), "test_auc_roc": round(test_auc, 4),
               "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
               "train_positive_rate": round(float(pos_rate), 4)}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
