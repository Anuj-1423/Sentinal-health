# Sentinel Health CKD Architecture

The local app has four layers:

1. **Data contract** (`sentinel_health/ckd_data.py`): robust ARFF parsing, label/schema validation, missing-value normalization, feature creation, and profiling.
2. **Training** (`sentinel_health/ckd_training.py`): fold-fitted preprocessing, 30 candidate configurations, duplicate-aware stratified CV, calibration, validation threshold selection, test evaluation, metrics, and plots.
3. **Application** (`app.py`): Flask dashboard pages for overview, EDA, data quality, demo scoring, model lab, explainability, observability, and registry.
4. **Inference** (`sentinel_health/api.py`): local FastAPI endpoint, model registry loading, input validation, aggregate telemetry, and Prometheus metrics.

Training writes artifacts to `artifacts/ckd/`; the active registry only changes when CKD training completes successfully.

This is a local educational prototype. No Docker or CI/CD workflow is included. No real patient data should be entered or logged.
