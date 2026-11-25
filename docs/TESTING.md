# Local test strategy

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q sentinel_health app.py run_training.py
.\.venv\Scripts\python.exe -m pytest -q tests -p no:cacheprovider
```

Tests cover target mapping and feature contract, patient-group partition isolation, and numeric/categorical drift calculations. Before extending the system, add model artifact round-trip, calibration/threshold isolation, endpoint schema/validation, SQLite telemetry privacy, subgroup minimum sample behavior, and dashboard page smoke tests.

Use metrics with uncertainty and cohort sizes, not fixed accuracy claims. The test partition is held out from candidate and threshold selection. Passing software tests is not clinical validation.
