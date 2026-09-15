"""FastAPI inference service for the sepsis early-warning model.

Run:  uvicorn serving.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import FEATURE_COLS

MODEL_DIR = Path(__file__).parent.parent / "models"

app = FastAPI(
    title="Sepsis Early-Warning API",
    description="72-hour sepsis onset risk scoring from ICU vitals/labs windows.",
    version="1.0.0",
)

_bundle = None
_explainer = None


def _load():
    global _bundle, _explainer
    if _bundle is None:
        _bundle = joblib.load(MODEL_DIR / "sepsis_xgb.joblib")
        _explainer = shap.TreeExplainer(_bundle["model"])
    return _bundle, _explainer


class WindowFeatures(BaseModel):
    heart_rate: float
    map_bp: float
    resp_rate: float
    temp_c: float
    spo2: float
    lactate: float
    wbc: float
    creatinine: float
    platelets: float
    fluid_balance_ml: float
    heart_rate_delta: float = 0.0
    map_bp_delta: float = 0.0
    resp_rate_delta: float = 0.0
    temp_c_delta: float = 0.0
    spo2_delta: float = 0.0
    lactate_delta: float = 0.0
    wbc_delta: float = 0.0
    creatinine_delta: float = 0.0
    platelets_delta: float = 0.0
    fluid_balance_ml_delta: float = 0.0
    age: float
    comorbidity_index: int
    frailty_index: float
    vasopressor_flag: int = 0
    hours_since_admission: float


class PredictionResponse(BaseModel):
    sepsis_onset_24h_probability: float
    risk_tier: str


class ExplainResponse(BaseModel):
    sepsis_onset_24h_probability: float
    risk_tier: str
    top_contributing_factors: List[dict]


def _risk_tier(p: float) -> str:
    if p >= 0.5:
        return "HIGH"
    if p >= 0.2:
        return "ELEVATED"
    return "LOW"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(window: WindowFeatures):
    bundle, _ = _load()
    x = pd.DataFrame([window.dict()])[FEATURE_COLS]
    p = float(bundle["model"].predict_proba(x)[0, 1])
    return PredictionResponse(sepsis_onset_24h_probability=round(p, 4), risk_tier=_risk_tier(p))


@app.post("/explain", response_model=ExplainResponse)
def explain(window: WindowFeatures):
    bundle, explainer = _load()
    x = pd.DataFrame([window.dict()])[FEATURE_COLS]
    p = float(bundle["model"].predict_proba(x)[0, 1])
    sv = explainer.shap_values(x)
    contributions = sorted(zip(FEATURE_COLS, sv[0]), key=lambda t: abs(t[1]), reverse=True)[:5]
    return ExplainResponse(
        sepsis_onset_24h_probability=round(p, 4),
        risk_tier=_risk_tier(p),
        top_contributing_factors=[{"feature": f, "shap_value": round(float(v), 4)} for f, v in contributions],
    )
