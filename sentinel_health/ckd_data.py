"""Load, validate, profile, and prepare the supplied UCI CKD ARFF data."""
from __future__ import annotations

import csv
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "chronic_kidney_disease_full.arff"
TARGET = "class"
POSITIVE_LABEL = "ckd"
NUMERIC_COLUMNS = ["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]
CATEGORICAL_COLUMNS = ["sg","al","su","rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
EXPECTED_COLUMNS = NUMERIC_COLUMNS[:2] + ["sg","al","su","rbc","pc","pcc","ba"] + NUMERIC_COLUMNS[2:] + ["htn","dm","cad","appet","pe","ane",TARGET]
FEATURE_COLUMNS = [c for c in EXPECTED_COLUMNS if c != TARGET]


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read ARFF rows robustly, normalizing '?' and correcting the source's one extra blank cell."""
    names: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not in_data:
                if stripped.lower() == "@data":
                    in_data = True
                    continue
                match = re.match(r"\s*@attribute\s+['\"]?([^'\"\s]+)['\"]?\s+", line, re.I)
                if match:
                    names.append(match.group(1))
                continue
            if not stripped or stripped.startswith("%"):
                continue
            row = [item.strip().strip("'").strip('"') for item in next(csv.reader([line], skipinitialspace=True))]
            # The distributed ARFF has one row with an extra empty nominal cell and one trailing comma.
            if len(row) == len(names) + 1 and "" in row:
                row.remove("")
            if len(row) == len(names) + 1 and row[-1] == "":
                row.pop()
            if len(row) != len(names):
                raise ValueError(f"Malformed ARFF record at line {line_no}: expected {len(names)} fields, got {len(row)}")
            rows.append(row)
    if names != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected CKD schema: {names}")
    df = pd.DataFrame(rows, columns=names).replace({"?": pd.NA, "": pd.NA})
    if df.empty or df[TARGET].isna().any():
        raise ValueError("CKD data is empty or has missing outcome labels.")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string").str.strip().str.lower().replace({"?": pd.NA,"": pd.NA})
    df[TARGET] = df[TARGET].astype("string").str.strip().str.lower()
    if set(df[TARGET].dropna().unique()) != {"ckd","notckd"}:
        raise ValueError(f"Unexpected CKD labels: {sorted(df[TARGET].dropna().unique())}")
    df["target"] = (df[TARGET] == POSITIVE_LABEL).astype("int8")
    return df


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build safe numeric/categorical model inputs and a small lab-derived ratio feature."""
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing CKD fields: {missing}")
    X = frame[FEATURE_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X["urea_creatinine_ratio"] = X["bu"] / X["sc"].where(X["sc"] > 0)
    X["lab_missing_count"] = X[NUMERIC_COLUMNS].isna().sum(axis=1).astype(float)
    for col in CATEGORICAL_COLUMNS:
        values=X[col].astype("string").str.strip().str.lower().replace({"?": pd.NA,"": pd.NA})
        X[col]=values.astype(object).where(values.notna(),np.nan)
    return X


def profile_dataset(df: pd.DataFrame) -> dict:
    missing = df[EXPECTED_COLUMNS].isna().sum().sort_values(ascending=False)
    counts = df[TARGET].value_counts().to_dict()
    return {"rows":int(len(df)),"columns":int(len(EXPECTED_COLUMNS)),"duplicate_rows":int(df.duplicated(subset=EXPECTED_COLUMNS).sum()),
        "target_counts":{str(k):int(v) for k,v in counts.items()},"target_rates":{str(k):float(v/len(df)) for k,v in counts.items()},
        "positive_label":POSITIVE_LABEL,"positive_rate":float(df.target.mean()),
        "missing_by_column":{str(k):int(v) for k,v in missing.items()},
        "missing_rate_by_column":{str(k):float(v/len(df)) for k,v in missing.items()},
        "numeric_columns":NUMERIC_COLUMNS,"categorical_columns":CATEGORICAL_COLUMNS}
