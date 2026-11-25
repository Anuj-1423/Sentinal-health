# Sentinel Health CKD Observability

## Training and data signals

The run records data schema/counts, target prevalence, duplicates, feature missingness, outlier review, split counts, 30 candidate CV scores, selected configuration, threshold policies, model version, fit duration, and holdout metrics. Imputers/encoders/scalers are fold-fitted.

## Model quality

The holdout tracks ROC-AUC, average precision, precision, recall, F1, accuracy, balanced accuracy, Brier score, alert rate, confusion matrix, and calibration. The report also contains a validation-selected higher-recall policy and local inference p50/p95/p99 latency. The 400-record benchmark is too small to treat one holdout as stable or externally validated.

## Drift and service

`feature_drift.csv` compares training and held-out feature distributions as an offline reference, not live drift. FastAPI `/health` and `/metrics` expose readiness and request metrics. The local event store contains aggregate score, threshold, class, model version, latency, source, and status; it does not store raw patient fields.

## Safety

This project predicts labels from a small historical source dataset. It is not validated for clinical screening or care. Never submit real patient data. Define prediction timing and verify feature availability before interpreting performance as future risk prediction.
