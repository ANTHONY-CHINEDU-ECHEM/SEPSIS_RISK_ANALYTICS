"""Sanity tests for the data -> features -> train -> predict pipeline."""
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import build_feature_matrix, FEATURE_COLS  # noqa: E402

DATA_PATH = ROOT / "data" / "icu_stays.csv"
MODEL_PATH = ROOT / "models" / "sepsis_xgb.joblib"


def test_data_exists_and_is_shaped_correctly():
    assert DATA_PATH.exists(), "Run data/generate_data.py first"
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 1000
    assert "sepsis_onset_24h" in df.columns
    assert df["sepsis_onset_24h"].isin([0, 1]).all()


def test_positive_rate_is_clinically_plausible():
    df = pd.read_csv(DATA_PATH)
    rate = df["sepsis_onset_24h"].mean()
    assert 0.03 < rate < 0.20, f"Positive rate {rate:.3%} outside plausible ICU sepsis-incidence range"


def test_feature_matrix_builds():
    df = pd.read_csv(DATA_PATH)
    X, y, groups, full = build_feature_matrix(df)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(y) == len(groups)
    assert not X.isnull().any().any()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run src/train.py first")
def test_model_predicts_valid_probabilities():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    df = pd.read_csv(DATA_PATH)
    X, y, groups, full = build_feature_matrix(df)
    preds = model.predict_proba(X.head(50))[:, 1]
    assert (preds >= 0).all() and (preds <= 1).all()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run src/train.py first")
def test_model_beats_random_baseline_auc():
    from sklearn.metrics import roc_auc_score
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    df = pd.read_csv(DATA_PATH)
    X, y, groups, full = build_feature_matrix(df)
    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)
    assert auc > 0.70, f"AUC {auc:.3f} is not meaningfully better than chance"
