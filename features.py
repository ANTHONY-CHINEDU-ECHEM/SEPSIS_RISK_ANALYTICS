"""Rolling-window feature engineering for the sepsis early-warning model.

For each observation window, adds trend features (delta vs. the previous
window) within the same ICU stay — a lightweight stand-in for the temporal
context a sequence model (LSTM/TFT) would learn implicitly.
"""
import pandas as pd

VITAL_COLS = ["heart_rate", "map_bp", "resp_rate", "temp_c", "spo2",
              "lactate", "wbc", "creatinine", "platelets", "fluid_balance_ml"]

STATIC_COLS = ["age", "comorbidity_index", "frailty_index"]

FEATURE_COLS = VITAL_COLS + [f"{c}_delta" for c in VITAL_COLS] + STATIC_COLS + \
               ["vasopressor_flag", "hours_since_admission"]


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stay_id", "window_idx"]).copy()
    for c in VITAL_COLS:
        df[f"{c}_delta"] = df.groupby("stay_id")[c].diff().fillna(0.0)
    return df


def build_feature_matrix(df: pd.DataFrame):
    df = add_trend_features(df)
    X = df[FEATURE_COLS].copy()
    y = df["sepsis_onset_24h"].copy()
    groups = df["stay_id"].copy()
    return X, y, groups, df
