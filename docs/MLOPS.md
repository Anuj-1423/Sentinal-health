# Sentinel Health CKD MLOps Notes

`run_training.py` profiles the source, evaluates 30 model/preprocessing configurations with 5-fold stratified CV, calibrates the selected estimator, selects thresholds on validation, and evaluates an untouched holdout. It writes the fitted model and metadata to `artifacts/ckd/`. The local JSON registry points to the selected CKD model. The bundle includes fitted preprocessing, estimator, probability calibration, threshold, version, and input schema.

`app.py` and the local FastAPI API load that registry artifact. Aggregate request metrics and latency are stored in local SQLite. Retraining and registry updates are manual; there is no automated promotion, Docker, or CI/CD.

The candidate score is internal to a small 400-row benchmark. Repeated cross-validation and a high holdout score do not replace external or prospective validation.
