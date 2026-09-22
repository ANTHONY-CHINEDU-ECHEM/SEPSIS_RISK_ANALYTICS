# 72 Hour Sepsis Onset Prediction from ICU Time Series Streams

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Reference Implementation](https://img.shields.io/badge/status-reference%20implementation-orange.svg)]()

**A temporal early warning system that predicts sepsis onset probability at 6, 12, 24, 48, and 72 hour horizons from rolling windows of ICU vitals and labs, built to outperform NEWS2 and qSOFA style track and trigger scoring on lead time.**

> **Reference implementation note.** The portfolio briefing specifies a Temporal Fusion Transformer trained on MIMIC IV. This repository ships a lightweight, fully runnable XGBoost on rolling windows reference implementation, trained on a synthetic ICU cohort, requiring no PhysioNet data use agreement to run, so the full pipeline, from data through features, model, explainability, and API, works out of the box on a laptop in under a minute. Swapping in real MIMIC IV or eICU data and a PyTorch based Temporal Fusion Transformer is a drop in extension; see [`docs/EXTENDING.md`](docs/EXTENDING.md).

---

## Table of Contents

1. [Overview](#overview)
2. [Rationale and Clinical Business Value](#rationale-and-clinical-business-value)
3. [Quick Start](#quick-start)
4. [What Is Inside](#what-is-inside)
5. [Technical Approach](#technical-approach)
6. [Evaluation Methodology](#evaluation-methodology)
7. [Explainability](#explainability)
8. [Serving and the Clinician Dashboard](#serving-and-the-clinician-dashboard)
9. [Model Card](#model-card)
10. [Known Limitations and Path to Production](#known-limitations-and-path-to-production)
11. [License](#license)

---

## Overview

Sepsis is a time critical medical emergency in which the body's response to infection injures its own tissues and organs. Left unrecognized, it can progress rapidly from early, subtle physiological changes to septic shock, organ failure, and death, often over a window of hours rather than days. The central clinical challenge is not diagnosing sepsis once it is obvious; it is recognizing the trajectory toward sepsis early enough that a clinical team can intervene before the patient deteriorates. This repository is a reference implementation of a system built to close that recognition gap: a model that continuously scores every ICU patient's short term sepsis risk from their existing vitals and lab stream, at multiple lookahead horizons, so that a rising risk trajectory becomes visible to the care team well before conventional bedside scoring would flag it.

---

## Rationale and Clinical Business Value

### Why Lead Time Is the Metric That Matters

Widely cited critical care research on septic shock has consistently found that the timeliness of effective treatment, most notably appropriate antimicrobial therapy, is one of the strongest modifiable determinants of survival, with mortality risk climbing as recognition and treatment are delayed hour by hour after onset. This is the clinical fact this entire project is organized around: a sepsis prediction system is not primarily being judged on whether it can eventually recognize sepsis, since by the time florid septic shock is present, a bedside nurse or a simple rule based score can recognize it too. It is being judged on how many additional hours of lead time it can buy a care team over the tools already in routine use, because those additional hours are what translate directly into a narrower and more treatable window of illness.

This is also why the title of this project frames its goal explicitly as outperforming NEWS2 and qSOFA style track and trigger scoring on lead time, rather than simply on a headline accuracy number. NEWS2 and qSOFA are deliberately simple, bedside calculable scores, built to be usable without a computer, and that simplicity is exactly what limits their sensitivity to the kind of gradual, multivariate physiological drift that often precedes sepsis. A model with access to the full rolling window of vitals and labs, rather than a handful of thresholded values at a single point in time, has the opportunity to detect that drift earlier, which is the entire commercial and clinical argument for building it in the first place.

### Why This Is Framed as a Business Problem, Not Only a Clinical One

Sepsis carries a disproportionate share of both inpatient mortality and inpatient cost relative to its prevalence, driven by prolonged ICU length of stay, escalation to higher acuity care, and the downstream cost of treating organ dysfunction that earlier intervention might have prevented. Every hour that recognition is delayed does not only carry clinical risk; it carries a direct, quantifiable cost to the health system in the form of additional ICU bed days, additional interventions, and, when an unrecognized case progresses to septic shock, a meaningfully higher probability of a long, complex, and expensive recovery, or no recovery at all. A hospital evaluating a system like this one is therefore evaluating two linked questions at once: does earlier recognition improve patient outcomes, and does it also reduce the average cost of a sepsis episode by shortening the acute, resource intensive phase of the illness. Both questions point in the same direction, which is part of why sepsis early warning is one of the more commercially mature use cases for predictive analytics in acute care.

### An Illustrative Unit Economics Model

As with clinical outcomes, the figures below are an illustrative model built from assumptions stated explicitly, intended to show the shape of the business case a hospital system would build before a pilot, not a measured result from a live deployment. This reference implementation has not been validated on real patient data, and none of these numbers should be read as claims about this specific model's real world performance; see [Known Limitations and Path to Production](#known-limitations-and-path-to-production) below.

| Quantity | Illustrative value |
|---|---|
| Annual ICU admissions at a mid sized hospital system | 12,000 |
| Share of ICU admissions that develop sepsis during their stay | 10 percent, or 1,200 cases per year |
| Average additional ICU cost attributable to a case recognized late, versus one recognized early | $8,000 per case (illustrative) |
| Share of the 1,200 annual cases where earlier model driven recognition plausibly shifts a late recognition into an early one | 20 percent, or 240 cases per year (illustrative, pending pilot validation) |
| Illustrative annual cost avoidance from earlier recognition (240 cases times $8,000) | $1,920,000 per year |

This is presented as a mechanism for reasoning about value, in the same spirit as the unit economics discussion in this author's other reference implementations, not as a validated return on investment. A real pilot would need to measure actual local sepsis incidence, actual current time to recognition, and actual cost per case at a specific institution before any of these figures could be treated as reliable.

### Why a Synthetic, XGBoost Based Reference Implementation Comes Before a Real Data, Transformer Based Production Model

The portfolio briefing behind this project specifies a Temporal Fusion Transformer trained on MIMIC IV, a real, de identified ICU dataset that requires a signed data use agreement with PhysioNet to access. That is the right eventual architecture and the right eventual data source for a production system, but it is the wrong starting point for a reference implementation meant to be cloned and run end to end by anyone evaluating this work, since it would require every reviewer to first obtain credentialed access to a restricted medical dataset before they could see the pipeline run at all. This repository instead ships a synthetic, MIMIC IV shaped cohort and a lightweight XGBoost model trained on rolling window features, so that the complete pipeline, from data generation through training, evaluation, explainability, and a live API, runs on a laptop in under a minute with no credentials and no cost.

This tradeoff is made explicit rather than hidden: gradient boosted trees over hand engineered rolling window features are a reasonable, fast, interpretable baseline, but they are not expected to match the sequence modeling capacity of a properly trained Temporal Fusion Transformer over real, high resolution ICU telemetry. The architecture is deliberately built so that upgrading the model and the data source later is a drop in extension rather than a rewrite; see [`docs/EXTENDING.md`](docs/EXTENDING.md) for the specific migration path.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/generate_data.py          # builds data/icu_stays.csv (synthetic)
python src/train.py                   # trains the model -> models/sepsis_xgb.joblib
python src/evaluate.py                # AUC-ROC, calibration, decision-curve report
uvicorn serving.api:app --reload      # POST /predict, /explain
```

---

## What Is Inside

| Path | Purpose |
|---|---|
| `data/generate_data.py` | Synthesizes a MIMIC IV shaped ICU cohort, including vitals, labs, and the sepsis onset label |
| `data/icu_stays.csv` | The generated dataset (approximately 50,000 windows), included so the repository runs standalone |
| `src/features.py` | Rolling window feature engineering, using 6 hour windows |
| `src/train.py` | Trains the XGBoost early warning classifier, using patient grouped cross validation |
| `src/evaluate.py` | Computes AUC ROC, sensitivity at a given specificity, the Brier score, and a decision curve plot |
| `src/explain.py` | SHAP based explainability, producing a per prediction contributing factors report |
| `serving/api.py` | A FastAPI service, exposing `/predict`, `/explain`, and `/health` |
| `models/` | The trained model artifact, plus `metrics.json` (checked in) |
| `tests/test_pipeline.py` | Sanity tests covering the data, training, and prediction path |
| `notebooks/eda.ipynb` | Exploratory data analysis |
| `dashboard/app.py` | A Streamlit clinician facing risk dashboard, including a SHAP waterfall view |

---

## Technical Approach

### Data Generation

`data/generate_data.py` produces a synthetic cohort shaped to resemble the structure of MIMIC IV ICU stays: rolling vitals (such as heart rate, respiratory rate, blood pressure, and temperature), rolling labs (such as white blood cell count and lactate), and a sepsis onset label anchored to a specific point in each simulated stay. Because the cohort is synthetic, it can be regenerated at will, shared without any data use agreement, and used to validate that the full pipeline behaves correctly before any real, credentialed clinical dataset is introduced.

### Feature Engineering

`src/features.py` builds rolling window features over 6 hour windows of the raw vitals and labs stream. This windowing choice reflects the clinical reality that a single vitals reading is noisy and can be misleading in isolation, while a short rolling trend, for example a steadily rising heart rate combined with a falling blood pressure over several hours, is a much stronger signal of physiological deterioration than any one reading taken alone.

### Model Training

`src/train.py` trains an XGBoost classifier over the engineered rolling window features, using patient grouped cross validation so that windows from the same patient stay never appear in both the training and validation folds at once. This is a deliberate methodological choice: because consecutive windows from the same ICU stay are highly correlated with each other, a naive random split would let the model partially memorize a specific patient's trajectory rather than learn a generalizable pattern, producing an evaluation score that looks better than the model would actually perform on a genuinely new patient.

---

## Evaluation Methodology

`src/evaluate.py` reports:

- **AUC ROC**, the model's overall discriminative ability across all prediction horizons.
- **Sensitivity at a fixed specificity**, since in a clinical alerting context the operationally relevant question is usually "how many true sepsis cases does the system catch, once the false alarm rate has been fixed at a level clinicians can tolerate," rather than a single threshold independent summary statistic.
- **The Brier score**, which evaluates how well calibrated the predicted probabilities are, not just how well they rank cases, which matters directly for a clinical tool where the number itself, not only its rank order, is shown to a bedside clinician.
- **A decision curve analysis**, which evaluates the model across a range of plausible clinical decision thresholds and compares the net benefit of using the model against the net benefit of treating everyone or treating no one, a standard methodology for judging whether a clinical prediction model is actually worth acting on at realistic operating points, rather than only being statistically better than chance.

---

## Explainability

`src/explain.py` uses SHAP to produce a per prediction, per patient contributing factors report: for any single risk score the model outputs, this shows which specific vitals and lab trends pushed that score up or down, and by how much. This exists for a specific operational reason rather than as a generic transparency feature: a bedside clinician is not going to act on a bare probability number they cannot interrogate, and a system that cannot explain why it flagged a particular patient is a system clinicians will learn to override or ignore. The `/explain` endpoint and the SHAP waterfall view in the Streamlit dashboard exist specifically to keep the model's reasoning inspectable at the point of care.

---

## Serving and the Clinician Dashboard

`serving/api.py` exposes a FastAPI service with three endpoints: `/predict`, which returns the sepsis onset probability at each of the 6, 12, 24, 48, and 72 hour horizons for a given patient window; `/explain`, which returns the SHAP based contributing factors behind a specific prediction; and `/health`, a standard service health check suitable for use in a container orchestration liveness probe.

`dashboard/app.py` provides a Streamlit based, clinician facing view of the same underlying model, intended for a pre rounds or handoff style workflow rather than for engineers: a ranked list of current ICU patients by rising risk trajectory, with a SHAP waterfall chart available for any patient a clinician wants to interrogate further before deciding whether to escalate.

---

## Model Card

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for intended use, limitations, and the fairness audit checklist referenced in the portfolio briefing. The model card is the authoritative source for what this model is and is not validated to do, and should be read before this repository's pipeline is adapted for anything beyond demonstration and reference purposes.

---

## Known Limitations and Path to Production

- **Synthetic training data.** The cohort used to train and evaluate the shipped model is synthetic, generated to resemble the structure of MIMIC IV rather than drawn from real patients. No performance figure produced by this reference implementation should be interpreted as a claim about real world clinical performance.
- **A gradient boosted tree baseline, not the specified Temporal Fusion Transformer.** XGBoost over rolling window features is a fast, interpretable, and easy to reproduce baseline, but it is not expected to match the sequence modeling capacity of a properly trained transformer architecture over real, high resolution ICU telemetry. See [`docs/EXTENDING.md`](docs/EXTENDING.md) for the intended migration path to real MIMIC IV or eICU data and a PyTorch based Temporal Fusion Transformer.
- **No clinical validation.** This repository has not been evaluated against real patient outcomes, has not undergone any regulatory review, and is not a certified medical device. It is a reference implementation intended to demonstrate architecture, methodology, and engineering practice.
- **A fairness audit is specified but requires real, demographically representative data to execute meaningfully.** The checklist in the model card exists so that a team moving this system toward production has a concrete starting point for that work; it cannot be meaningfully completed against synthetic data alone.

---

## License

This project is distributed under the MIT License, for portfolio and demonstration use. It is not a certified medical device and is not intended for clinical deployment without regulatory review.
