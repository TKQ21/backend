"""Versioned model registry + persistence.

Every successful training run writes a new immutable version folder:

    artifacts/models/v<N>/model.pkl
    artifacts/models/v<N>/metadata.json

`artifacts/models/index.json` records the version list and which version is
currently deployed for inference. Nothing here is simulated: the served model
is the exact estimator object fitted by train_ml.run_pipeline.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import joblib

ARTIFACTS = Path(__file__).parent / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
INDEX = MODELS_DIR / "index.json"

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"version": None, "model": None, "meta": None}


def dataset_id(content: bytes) -> str:
    return "ds-" + hashlib.sha256(content).hexdigest()[:16]


def _read_index() -> dict[str, Any]:
    try:
        return json.loads(INDEX.read_text())
    except Exception:
        return {"current": None, "versions": []}


def _write_index(idx: dict[str, Any]) -> None:
    INDEX.write_text(json.dumps(idx, indent=2, default=str))


def register(estimator: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Persist the estimator as the next version and mark it as deployed."""
    with _LOCK:
        idx = _read_index()
        version = max([int(v["version"]) for v in idx["versions"]], default=0) + 1
        folder = MODELS_DIR / f"v{version}"
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, folder / "model.pkl")
        meta = {
            "version": version,
            "modelName": metadata.get("modelName"),
            "family": metadata.get("family"),
            "trainedAt": metadata.get("trainedAt") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trainedAtEpoch": time.time(),
            "runId": metadata.get("runId"),
            "datasetId": metadata.get("datasetId"),
            "datasetRows": metadata.get("datasetRows"),
            "dateColumn": metadata.get("dateColumn"),
            "targetColumn": metadata.get("targetColumn"),
            "features": metadata.get("features", []),
            "params": metadata.get("params", {}),
            "metrics": metadata.get("metrics", {}),
            "mlflowVersion": metadata.get("mlflowVersion"),
            "mlflowRunId": metadata.get("mlflowRunId"),
            "artifactPath": str(folder / "model.pkl"),
        }
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        idx["versions"] = [v for v in idx["versions"] if int(v["version"]) != version]
        idx["versions"].append({
            "version": version,
            "modelName": meta["modelName"],
            "trainedAt": meta["trainedAt"],
            "runId": meta["runId"],
            "datasetId": meta["datasetId"],
            "testRmse": (meta["metrics"].get("test") or {}).get("rmse"),
            "testR2": (meta["metrics"].get("test") or {}).get("r2"),
        })
        idx["versions"].sort(key=lambda v: -int(v["version"]))
        idx["current"] = version
        _write_index(idx)
        _CACHE.update(version=None, model=None, meta=None)
        return meta


def _meta_path(version: int) -> Path:
    return MODELS_DIR / f"v{version}" / "metadata.json"


def current_version() -> int | None:
    v = _read_index().get("current")
    return int(v) if v is not None else None


def get_metadata(version: int | None = None) -> dict[str, Any] | None:
    version = version if version is not None else current_version()
    if version is None:
        return None
    p = _meta_path(version)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_versions() -> dict[str, Any]:
    idx = _read_index()
    return {"current": idx.get("current"), "versions": idx.get("versions", [])}


def activate(version: int) -> dict[str, Any]:
    if not _meta_path(version).exists():
        raise FileNotFoundError(f"model version v{version} does not exist")
    with _LOCK:
        idx = _read_index()
        idx["current"] = int(version)
        _write_index(idx)
        _CACHE.update(version=None, model=None, meta=None)
    return list_versions()


def load_current() -> tuple[Any, dict[str, Any]]:
    """Load (and cache) the deployed model for inference."""
    version = current_version()
    if version is None:
        raise FileNotFoundError("No model has been registered yet. Train a model first.")
    with _LOCK:
        if _CACHE["version"] == version and _CACHE["model"] is not None:
            return _CACHE["model"], _CACHE["meta"]
        model = joblib.load(MODELS_DIR / f"v{version}" / "model.pkl")
        meta = get_metadata(version) or {}
        _CACHE.update(version=version, model=model, meta=meta)
        return model, meta
