"""Local CKD classification API for an educational benchmark."""
from __future__ import annotations
import math,time
from functools import lru_cache
from typing import Any
import joblib
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from prometheus_client import Counter,Histogram,generate_latest,CONTENT_TYPE_LATEST
from starlette.responses import Response
from .observability import record_prediction
from .registry import active_model_path

app=FastAPI(title="Sentinel Health CKD API",version="2.0.0",description="Educational CKD dataset classification; not for clinical use.")
REQUESTS=Counter("sentinel_health_ckd_requests_total","CKD prediction requests",["status"])
LATENCY=Histogram("sentinel_health_ckd_latency_seconds","CKD prediction latency")
@lru_cache(maxsize=2)
def _load_model(path:str,mtime:float):return joblib.load(path)
def _model():
    path=active_model_path()
    if not path.exists():raise FileNotFoundError("No CKD model exists; run run_training.py first.")
    return _load_model(str(path.resolve()),path.stat().st_mtime)
class PredictionRequest(BaseModel):
    patient:dict[str,Any]
@app.get("/health")
def health():
    try:_model();ready=True
    except Exception:ready=False
    return {"status":"ok" if ready else "degraded","model_loaded":ready,"service":"sentinel-health-ckd-api","use":"educational only"}
@app.get("/metrics")
def metrics():return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)
@app.post("/api/v1/predict")
def predict(request:PredictionRequest):
    started=time.perf_counter();version="unavailable"
    try:
        model=_model();version=model.model_version
        for value in request.patient.values():
            if value is None:continue
            try:number=float(value)
            except (TypeError,ValueError):continue
            if math.isinf(number):raise ValueError("Infinite numeric input is invalid; omit unknown values or pass null.")
        import pandas as pd
        frame=pd.DataFrame([request.patient]);prob=float(model.predict_proba(frame)[0,1]);threshold=float(model.threshold)
        predicted="ckd" if prob>=threshold else "notckd";elapsed=(time.perf_counter()-started)*1000
        event_id=record_prediction(model_version=version,probability=prob,threshold=threshold,risk_band=predicted,latency_ms=elapsed,source="api")
        REQUESTS.labels(status="success").inc();LATENCY.observe(elapsed/1000)
        return {"predicted_class":predicted,"ckd_probability":prob,"threshold":threshold,"model_version":version,
            "event_id":event_id,"latency_ms":elapsed,"notice":"Educational dataset output only; not a diagnosis or clinical recommendation."}
    except (ValueError,FileNotFoundError) as exc:
        elapsed=(time.perf_counter()-started)*1000;record_prediction(model_version=version,probability=None,threshold=None,risk_band=None,latency_ms=elapsed,source="api",success=False,error_type=type(exc).__name__);REQUESTS.labels(status="error").inc();LATENCY.observe(elapsed/1000)
        raise HTTPException(422 if isinstance(exc,ValueError) else 503,detail=str(exc)) from exc
    except Exception as exc:
        elapsed=(time.perf_counter()-started)*1000;record_prediction(model_version=version,probability=None,threshold=None,risk_band=None,latency_ms=elapsed,source="api",success=False,error_type=type(exc).__name__);REQUESTS.labels(status="error").inc();LATENCY.observe(elapsed/1000)
        raise HTTPException(500,detail="Prediction failed; inspect local service logs.") from exc
