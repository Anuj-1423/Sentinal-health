"""Local, privacy-minimized prediction telemetry for Sentinel Health."""
from __future__ import annotations
from datetime import datetime, timezone
import logging, os, sqlite3, tempfile
from pathlib import Path
from typing import Any
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
_VERCEL = os.getenv("VERCEL") == "1" or bool(os.getenv("VERCEL_ENV"))
_TMP_ROOT = Path(tempfile.gettempdir()).resolve()
_LOCAL_DB = ROOT/"artifacts"/"ckd"/"ckd_events.db"
_configured_db = os.getenv("SENTINEL_HEALTH_DB")
if _VERCEL:
    _candidate_db = Path(_configured_db).resolve() if _configured_db else None
    # Vercel's deployment bundle is read-only; keep SQLite telemetry in /tmp.
    DB_PATH = (_candidate_db if _candidate_db and _candidate_db.is_relative_to(_TMP_ROOT)
               else _TMP_ROOT/"sentinel-health"/"ckd_events.db")
else:
    DB_PATH=Path(_configured_db or str(_LOCAL_DB))

def _connect():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(DB_PATH,timeout=15)
    c.execute("""CREATE TABLE IF NOT EXISTS prediction_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
      model_version TEXT NOT NULL, probability REAL, threshold REAL,
      risk_band TEXT, latency_ms REAL NOT NULL, source TEXT NOT NULL, success INTEGER NOT NULL,
      error_type TEXT)""")
    c.commit(); return c

def log_prediction(*,model_version:str,probability:float|None,threshold:float|None,
                   risk_band:str|None,latency_ms:float,source:str,success:bool=True,
                   error_type:str|None=None)->int:
    with _connect() as c:
        cur=c.execute("INSERT INTO prediction_events(created_at,model_version,probability,threshold,risk_band,latency_ms,source,success,error_type) VALUES(?,?,?,?,?,?,?,?,?)",
          (datetime.now(timezone.utc).isoformat(),model_version,probability,threshold,risk_band,float(latency_ms),source,int(success),error_type))
        return int(cur.lastrowid)

def record_prediction(**kwargs)->int|None:
    """Record telemetry when possible without allowing storage to break scoring."""
    try:
        return log_prediction(**kwargs)
    except Exception:
        logging.getLogger(__name__).exception("Prediction telemetry could not be recorded")
        return None

def recent_events(limit:int=500)->pd.DataFrame:
    with _connect() as c:return pd.read_sql_query("SELECT * FROM prediction_events ORDER BY id DESC LIMIT ?",c,params=(int(limit),))

def telemetry_summary()->dict[str,Any]:
    events=recent_events(10000)
    if events.empty:return {"requests":0,"errors":0,"error_rate":0.,"p50_ms":None,"p95_ms":None,"p99_ms":None,"throughput_per_minute":0.}
    success=events[events.success.astype(bool)]; lat=success.latency_ms.astype(float)
    ts=pd.to_datetime(events.created_at,format="mixed",utc=True,errors="coerce").dropna()
    span=max((ts.max()-ts.min()).total_seconds(),60.) if len(ts)>1 else 60.
    return {"requests":int(len(events)),"errors":int((~events.success.astype(bool)).sum()),
        "error_rate":float((~events.success.astype(bool)).mean()),
        "p50_ms":float(lat.quantile(.5)) if len(lat) else None,
        "p95_ms":float(lat.quantile(.95)) if len(lat) else None,
        "p99_ms":float(lat.quantile(.99)) if len(lat) else None,
        "throughput_per_minute":float(len(events)/span*60)}
