# 72-Hour Sepsis Onset Prediction from ICU Time-Series Streams

A temporal early-warning system that predicts sepsis onset probability at
6/12/24/48/72-hour horizons from rolling windows of ICU vitals and labs —
built to outperform NEWS2/qSOFA-style track-and-trigger scoring on lead time.

> **Reference-implementation note.** The portfolio briefing specifies a
> Temporal Fusion Transformer trained on MIMIC-IV. This repo ships a
> lightweight, fully-runnable **XGBoost-on-rolling-windows** reference
> implementation trained on a **synthetic** ICU cohort (no PhysioNet DUA
> required to run it), so the full pipeline — data → features → model →
> explainability → API — works out of the box on a laptop in under a
> minute. Swapping in real MIMIC-IV/eICU data and a PyTorch TFT is a drop-in
> extension; see [`docs/EXTENDING.md`](docs/EXTENDING.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/generate_data.py          # builds data/icu_stays.csv (synthetic)
python src/train.py                   # trains the model -> models/sepsis_xgb.joblib
python src/evaluate.py                # AUC-ROC, calibration, decision-curve report
uvicorn serving.api:app --reload      # POST /predict, /explain
```

## What's inside

| Path | Purpose |
|---|---|
| `data/generate_data.py` | Synthesizes a MIMIC-IV-shaped ICU cohort (vitals, labs, sepsis-onset label) |
| `data/icu_stays.csv` | The generated dataset (~50,000 windows) — included so the repo runs standalone |
| `src/features.py` | Rolling-window feature engineering (6h windows) |
| `src/train.py` | Trains the XGBoost early-warning classifier with patient-grouped CV |
| `src/evaluate.py` | AUC-ROC, sensitivity@specificity, Brier score, decision-curve plot |
| `src/explain.py` | SHAP explainability — per-prediction contributing-factors report |
| `serving/api.py` | FastAPI service: `/predict`, `/explain`, `/health` |
| `models/` | Trained model artifact + metrics.json (checked in) |
| `tests/test_pipeline.py` | Sanity tests for the data → train → predict path |
| `notebooks/eda.ipynb` | Exploratory data analysis |
| `dashboard/app.py` | Streamlit clinician-facing risk dashboard (SHAP waterfall) |

## Model card

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for intended use, limitations,
and the fairness-audit checklist referenced in the portfolio briefing.

## License

MIT — for portfolio/demonstration use. Not a certified medical device;
not for clinical deployment without regulatory review.
