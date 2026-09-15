"""Full evaluation report: AUC-ROC, sensitivity@specificity, calibration
(Brier score), and a simple decision-curve summary, on the held-out test
split saved by train.py.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import build_feature_matrix

DATA_PATH = Path(__file__).parent.parent / "data" / "icu_stays.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"


def sensitivity_at_specificity(y_true, y_score, target_specificity=0.90):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - target_specificity))
    return float(tpr[idx]), float(thresh[idx]), float(specificity[idx])


def main():
    bundle = joblib.load(MODEL_DIR / "sepsis_xgb.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    df = pd.read_csv(DATA_PATH)
    X, y, groups, _ = build_feature_matrix(df)
    test_idx = np.load(MODEL_DIR / "test_idx.npy", allow_pickle=True)
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    sens, thresh, spec = sensitivity_at_specificity(y_test, y_score, 0.90)
    brier = brier_score_loss(y_test, y_score)

    report = {
        "test_auc_roc": round(float(auc), 4),
        "sensitivity_at_90pct_specificity": round(sens, 4),
        "operating_threshold": round(thresh, 4),
        "achieved_specificity": round(spec, 4),
        "brier_score": round(float(brier), 4),
        "n_test_windows": int(len(y_test)),
        "test_positive_rate": round(float(y_test.mean()), 4),
    }
    out_path = MODEL_DIR / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
