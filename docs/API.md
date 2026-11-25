# Sentinel Health CKD API

Start locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn sentinel_health.api:app --host 127.0.0.1 --port 8000
```

- `GET /health`: API and CKD model readiness.
- `GET /metrics`: Prometheus-compatible request and latency metrics.
- `POST /api/v1/predict`: returns predicted class, calibrated CKD probability, validation-selected threshold, model version, latency, and educational-use notice.

Request body: `{"patient": { ... }}` with the 24 input fields in `sentinel_health.ckd_data.FEATURE_COLUMNS`. Omit unknown values or pass `null`; the model imputers handle missing data. The endpoint rejects infinite numeric values and extra/missing input columns. It does not accept the target field.

This local demo has no authentication and is not suitable for clinical or public deployment. Never submit real patient data.
