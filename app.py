"""Streamlit clinician-facing risk dashboard.

Run: streamlit run dashboard/app.py
(streamlit is optional — not in requirements.txt by default to keep the
core pipeline lightweight; `pip install streamlit` to use this dashboard.)
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import build_feature_matrix, FEATURE_COLS

ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="Sepsis Early-Warning Dashboard", layout="wide")
st.title("🏥 Sepsis Early-Warning — Clinician Dashboard")

bundle = joblib.load(ROOT / "models" / "sepsis_xgb.joblib")
model = bundle["model"]
explainer = shap.TreeExplainer(model)

df = pd.read_csv(ROOT / "data" / "icu_stays.csv")
X, y, groups, full = build_feature_matrix(df)

stay_ids = sorted(full["stay_id"].unique())
stay_id = st.selectbox("Select ICU stay", stay_ids)

stay_rows = full[full["stay_id"] == stay_id].sort_values("window_idx")
st.line_chart(stay_rows.set_index("hours_since_admission")[["heart_rate", "map_bp", "lactate"]])

latest = stay_rows.iloc[[-1]]
x_latest = latest[FEATURE_COLS]
proba = float(model.predict_proba(x_latest)[0, 1])

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Sepsis-onset-24h risk", f"{proba:.1%}")
    tier = "🔴 HIGH" if proba >= 0.5 else ("🟠 ELEVATED" if proba >= 0.2 else "🟢 LOW")
    st.write(f"**Risk tier:** {tier}")

with col2:
    sv = explainer.shap_values(x_latest)
    contrib = sorted(zip(FEATURE_COLS, sv[0]), key=lambda t: abs(t[1]), reverse=True)[:6]
    st.write("**Top contributing factors:**")
    st.table(pd.DataFrame(contrib, columns=["feature", "shap_value"]))
