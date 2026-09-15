"""SHAP-based explainability: per-window contributing-factors report."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import build_feature_matrix

DATA_PATH = Path(__file__).parent.parent / "data" / "icu_stays.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"


def get_explainer():
    bundle = joblib.load(MODEL_DIR / "sepsis_xgb.joblib")
    return shap.TreeExplainer(bundle["model"]), bundle["feature_cols"]


def explain_row(x_row: pd.Series, top_k: int = 5):
    explainer, feature_cols = get_explainer()
    sv = explainer.shap_values(x_row[feature_cols].to_frame().T)
    contributions = list(zip(feature_cols, sv[0]))
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)
    return [{"feature": f, "shap_value": round(float(v), 4)} for f, v in contributions[:top_k]]


def main():
    df = pd.read_csv(DATA_PATH)
    X, y, groups, full = build_feature_matrix(df)
    # pick a genuine positive window to demonstrate
    pos_idx = full.index[full["sepsis_onset_24h"] == 1][0]
    row = X.loc[pos_idx]
    result = explain_row(row)
    print(f"Top contributing factors for window index {pos_idx} (true label = sepsis onset):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
