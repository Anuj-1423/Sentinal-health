"""Local model registry pointer for trained CKD models."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT/"artifacts"/"ckd"
REGISTRY=ARTIFACTS/"model_registry.json"
def list_versions():
    if not REGISTRY.exists():return {"active_version":None,"versions":[]}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))
def active_model_path()->Path:
    state=list_versions()
    for item in state.get("versions",[]):
        if item.get("version")==state.get("active_version"):
            p=Path(item["artifact"])
            if not p.is_absolute():p=ROOT/p
            if p.exists():return p
    return ARTIFACTS/"ckd_model.joblib"
