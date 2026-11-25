# Sentinel Health · CKD Classification

Sentinel Health is an educational machine-learning project for binary classification (`ckd` vs `notckd`) using the supplied UCI Chronic Kidney Disease ARFF data. The dashboard provides EDA, missingness and outlier review, a 30-configuration model comparison, threshold analysis, explainability, observability, local prediction, and an API.

> This 400-record historical benchmark is not prospectively or externally validated. Do not use its predictions for diagnosis, screening, treatment, or patient-care decisions. Do not enter real patient information.

## Dataset

Default input: `chronic_kidney_disease_full.arff`. The project also contains `chronic_kidney_disease.arff`; they represent the same 400 rows. The loader handles the source's `?` missing markers, one malformed extra blank field, and validates the 24 predictors plus class. Target counts are 250 `ckd` and 150 `notckd`.

## Set up and train (PowerShell)

```powershell
uv venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe run_training.py --data chronic_kidney_disease_full.arff --artifacts artifacts/ckd --cv-folds 5
```

The run screens 30 model/preprocessing configurations with stratified cross-validation. Preprocessing is fit inside CV folds. It calibrates the selected model, selects thresholds on validation, then evaluates on an untouched stratified holdout. Artifacts are saved under `artifacts/ckd/`.

## Latest local holdout result

Selected candidate: `random_forest_d4_leaf2_balanced`. Holdout size: 80 records (50 CKD, 30 not-CKD).

- ROC-AUC **0.996**, average precision **0.998**, accuracy **0.975**
- Precision **0.962**, recall **1.000**, F1 **0.980**, balanced accuracy **0.967**
- Brier score **0.021**; confusion matrix TN/FP/FN/TP: **28/2/0/50**
- Alternative validation 80%-recall threshold: holdout recall **0.920**, precision **1.000**

These are internal results on a very small dataset. The near-perfect cross-validation and holdout scores may reflect how strongly the supplied measurements align with the dataset's CKD label. They do not establish performance on new hospitals or future patients. Some inputs may be recorded during or after diagnosis; define the prediction point and exclude unavailable/downstream fields before interpreting this as prediction.

## Run the app and API

```powershell
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\python.exe -m uvicorn sentinel_health.api:app --host 127.0.0.1 --port 8000
```

Dashboard: `http://localhost:8501` (Flask)  
API docs: `http://localhost:8000/docs`  
Health: `http://localhost:8000/health`

## Notebook and artifacts

Open `data.ipynb` in VS Code using the `.venv` kernel and run cells in order. It covers `.head()`, `.tail()`, `.info()`, `.describe()`, EDA, missing values, outlier review, feature preparation, splits, all 30 candidates, held-out metrics, PR/ROC/calibration curves, feature importance, drift, and inference latency.

```text
app.py                      Flask CKD dashboard
templates/                  Responsive Jinja dashboard pages
static/                     Dashboard styling and local Plotly runtime
run_training.py             30-candidate CKD training CLI
sentinel_health/ckd_data.py  ARFF parsing, validation, features, profile
sentinel_health/ckd_training.py CV, calibration, metrics, artifacts
sentinel_health/api.py       Local CKD scoring API
artifacts/ckd/               Model, report, plots, registry, quality outputs
tests/test_ckd.py             CKD data and candidate checks
```
