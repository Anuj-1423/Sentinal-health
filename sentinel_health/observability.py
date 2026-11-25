"""Local, privacy-minimized prediction telemetry for Sentinel Health."""
from __future__ import annotations
from datetime import datetime, timezone
import logging, os, sqlite3, ssl, tempfile
from pathlib import Path
from typing import Any
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT.parent/".env")
except ImportError:
    env_file=ROOT.parent/".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key,separator,value=line.partition("=")
            if separator and key.strip() and not key.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(),value.strip().strip("\"'"))
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

_EVENT_COLUMNS = ["id", "created_at", "model_version", "probability", "threshold",
                  "risk_band", "latency_ms", "source", "success", "error_type"]

def _mysql_enabled()->bool:
    return all(os.getenv(key, "").strip() for key in
               ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD"))

def _mysql_connect():
    import pymysql
    from pymysql.cursors import DictCursor

    ssl_mode=os.getenv("MYSQL_SSL_MODE","VERIFY_IDENTITY").strip().upper()
    ca_value=os.getenv("MYSQL_SSL_CA","").strip()
    if ssl_mode not in {"VERIFY_CA","VERIFY_IDENTITY"}:
        raise ValueError("MYSQL_SSL_MODE must be VERIFY_CA or VERIFY_IDENTITY.")
    if not ca_value:
        raise ValueError("Set MYSQL_SSL_CA to the Aiven CA certificate path or PEM contents.")
    if "-----BEGIN CERTIFICATE-----" in ca_value:
        tls=ssl.create_default_context(cadata=ca_value)
    else:
        tls=ssl.create_default_context(cafile=ca_value)
    tls.check_hostname=(ssl_mode=="VERIFY_IDENTITY")
    connection = pymysql.connect(
        host=os.environ["MYSQL_HOST"], port=int(os.environ["MYSQL_PORT"]),
        user=os.environ["MYSQL_USER"], password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"], charset="utf8mb4",
        cursorclass=DictCursor, connect_timeout=8, read_timeout=8, write_timeout=8,
        ssl=tls,
    )
    with connection.cursor() as cursor:
        cursor.execute("""CREATE TABLE IF NOT EXISTS prediction_events (
          id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY, created_at VARCHAR(40) NOT NULL,
          model_version VARCHAR(255) NOT NULL, probability DOUBLE NULL, threshold DOUBLE NULL,
          risk_band VARCHAR(32) NULL, latency_ms DOUBLE NOT NULL, source VARCHAR(128) NOT NULL,
          success TINYINT(1) NOT NULL, error_type VARCHAR(128) NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    connection.commit()
    return connection

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
    values=(datetime.now(timezone.utc).isoformat(),model_version,probability,threshold,
            risk_band,float(latency_ms),source,int(success),error_type)
    if _mysql_enabled():
        connection=_mysql_connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO prediction_events(created_at,model_version,probability,threshold,risk_band,latency_ms,source,success,error_type) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",values)
                event_id=int(cursor.lastrowid)
            connection.commit()
            return event_id
        finally:
            connection.close()
    with _connect() as c:
        cur=c.execute("INSERT INTO prediction_events(created_at,model_version,probability,threshold,risk_band,latency_ms,source,success,error_type) VALUES(?,?,?,?,?,?,?,?,?)",values)
        return int(cur.lastrowid)

def record_prediction(**kwargs)->int|None:
    """Record telemetry when possible without allowing storage to break scoring."""
    try:
        return log_prediction(**kwargs)
    except Exception:
        logging.getLogger(__name__).exception("Prediction telemetry could not be recorded")
        return None

def recent_events(limit:int=500)->pd.DataFrame:
    if _mysql_enabled():
        connection=_mysql_connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM prediction_events ORDER BY id DESC LIMIT %s",(int(limit),))
                rows=cursor.fetchall()
            return pd.DataFrame(rows,columns=_EVENT_COLUMNS)
        finally:
            connection.close()
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
