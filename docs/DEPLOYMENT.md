# Sentinel Health CKD Repository Setup

## Structure

```text
app.py
templates/
static/
run_training.py
chronic_kidney_disease_full.arff
chronic_kidney_disease.arff
sentinel_health/ckd_data.py
sentinel_health/ckd_training.py
sentinel_health/api.py
sentinel_health/observability.py
sentinel_health/registry.py
data.ipynb
artifacts/ckd/
tests/test_ckd.py
```

The project uses a local virtual environment:

```powershell
uv venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

Train the 30 configurations and write outputs:

```powershell
.\.venv\Scripts\python.exe run_training.py --data chronic_kidney_disease_full.arff --artifacts artifacts/ckd --cv-folds 5
```

Run the dashboard and API in separate terminals:

```powershell
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\python.exe -m uvicorn sentinel_health.api:app --host 127.0.0.1 --port 8000
```

Checks:

```powershell
.\.venv\Scripts\python.exe -m compileall -q sentinel_health app.py run_training.py
.\.venv\Scripts\python.exe -m pytest -q tests
```

The benchmark and outputs are not clinical. Do not enter real patient data.
