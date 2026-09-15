# Extending this reference implementation

This repo ships a lightweight, fully-runnable version of the architecture
described in the portfolio briefing. Suggested upgrade path to the full
production design:

1. **Real data.** Sign a PhysioNet Data Use Agreement, pull MIMIC-IV /
   eICU, and reproduce the same window-level schema produced by
   `data/generate_data.py` (see column list in `src/features.py`).
2. **Sequence model.** Replace the XGBoost-on-engineered-windows model in
   `src/train.py` with a PyTorch Temporal Fusion Transformer
   (`pytorch-forecasting` provides a ready TFT implementation) that
   consumes the raw 6-hour sequences directly instead of hand-engineered
   deltas.
3. **Streaming pipeline.** Swap the batch CSV read for a Kafka/Pub-Sub
   consumer reading HL7v2/FHIR observation events, with a Redis feature
   store for the rolling windows (see the System Architecture slide in the
   briefing deck for the target design).
4. **Serving.** Containerize `serving/api.py` and deploy to a managed
   endpoint (Vertex AI / SageMaker); add the Airflow DAGs referenced in the
   briefing for scheduled retraining and backfill.
5. **Governance.** Wire in the fairness-audit checklist in
   `docs/MODEL_CARD.md` against real demographic fields before any
   clinical pilot.
