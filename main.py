"""FastAPI serving layer: CSV upload -> background training job -> polling -> results.

Production extras: API-key auth, rate limiting, SQLite job/run history,
automated retraining scheduler, monitoring + error alerting.

Run:  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import io
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import db
import drift
import monitoring
import performance
import registry
import retrain
import security
from preprocessing import FEATURES, detect_schema
from train_ml import ARTIFACTS, run_pipeline

app = FastAPI(title="ElectriPredict ML API", version="3.0.0")

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
RUNS_FILE = ARTIFACTS / "runs.json"
Auth = Depends(security.authenticate)


def _set(job_id: str, **kw: Any) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)
    try:
        db.upsert_job(job_id, **kw)
    except Exception as exc:
        monitoring.event("error", "db.upsert_job_failed", str(exc))


def _load_runs() -> list[dict[str, Any]]:
    rows = db.list_runs()
    if rows:
        return rows
    try:
        return json.loads(RUNS_FILE.read_text())
    except Exception:
        return []


def _append_run(payload: dict[str, Any]) -> None:
    run = {
        "runId": payload["runId"],
        "experiment": "electricity-demand-forecast",
        "finishedAt": payload["finishedAt"],
        "dataset": {
            "rows": payload["schema"]["usableRows"],
            "target": payload["schema"]["targetColumn"],
            "dateColumn": payload["schema"]["dateColumn"],
        },
        "bestModel": payload["best"]["name"],
        "metrics": {
            "val_rmse": payload["best"]["val"]["rmse"],
            "val_mae": payload["best"]["val"]["mae"],
            "val_r2": payload["best"]["val"]["r2"],
            "test_rmse": payload["best"]["test"]["rmse"],
            "test_mae": payload["best"]["test"]["mae"],
            "test_r2": payload["best"]["test"]["r2"],
            "test_mape": payload["best"]["test"]["mape"],
        },
        "registeredVersion": payload.get("registeredVersion"),
    }
    db.add_run(run)
    runs = [run] + [r for r in _load_runs() if r.get("runId") != run["runId"]]
    RUNS_FILE.write_text(json.dumps(runs[:50], indent=2, default=str))


def _sanitize(o: Any) -> Any:
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    return o


def _train_job(job_id: str, content: bytes, horizon: int, source: str = "upload") -> None:
    t0 = time.time()
    try:
        _set(job_id, status="running", progress=5, message="Parsing CSV")
        try:
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        except Exception as exc:
            raise ValueError(f"CSV could not be parsed: {exc}") from exc
        if df.empty or not len(df.columns):
            raise ValueError("CSV has no rows or no columns.")
        rep = detect_schema(df)
        if rep.errors:
            raise ValueError("; ".join(rep.errors))
        _set(job_id, schema=rep.dict())

        def progress(p: int, m: str) -> None:
            _set(job_id, progress=p, message=m)

        payload = run_pipeline(df, rep, horizon=horizon, run_id=job_id, progress=progress,
                               dataset_id=registry.dataset_id(content))
        _append_run(payload)
        _set(job_id, status="succeeded", progress=100, message="Done",
             result=_sanitize(payload), schema=payload["schema"])
        if source == "upload":
            retrain.snapshot_dataset(content)
        else:
            retrain._save(lastStatus="succeeded", failures=0)
        monitoring.event("info", "train.succeeded",
                         f"{job_id} best={payload['best']['name']} in {time.time() - t0:.1f}s")
    except Exception as exc:
        _set(job_id, status="failed", progress=100, message="Training failed", error=str(exc))
        if source != "upload":
            fails = (retrain._state().get("failures", 0)) + 1
            retrain._save(failures=fails, paused=fails >= retrain.MAX_FAILURES,
                          pauseReason=str(exc)[:300], lastStatus="failed")
        monitoring.event("error", "train.failed", f"{job_id}: {exc}")
    finally:
        del content


def _start_job(content: bytes, source: str, file_name: str = "scheduled.csv",
               horizon: int = 30) -> str:
    job_id = f"run-{uuid.uuid4().hex[:10]}"
    _set(job_id, status="queued", progress=0, message="Queued",
         fileName=file_name, source=source, createdAt=time.time())
    threading.Thread(target=_train_job, args=(job_id, content, horizon, source),
                     daemon=True).start()
    return job_id


@app.on_event("startup")
def _startup() -> None:
    db.init()
    monitoring.event("info", "service.started", f"auth={security.AUTH_ENABLED}")
    retrain.start(lambda content, source: _start_job(content, source))


@app.get("/api/ml/health")
def health() -> dict[str, Any]:
    meta_path = ARTIFACTS / "model.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return {
        "status": "ok",
        "service": "ElectriPredict ML API",
        "registered_model": (meta or {}).get("model", {}).get("name"),
        "registered_version": (meta or {}).get("registered_version"),
        "security": security.config(),
        "retraining": retrain.status(),
        "deployed_version": registry.current_version(),
        "deployed_model": (registry.get_metadata() or {}).get("modelName"),
    }


@app.post("/api/ml/train")
async def train(background: BackgroundTasks, file: UploadFile = File(...), horizon: int = 30,
                cid: str = Auth) -> dict[str, str]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    security.limit_training(cid)
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"CSV is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")
    job_id = _start_job(content, "upload", file.filename, max(1, min(365, horizon)))
    monitoring.event("info", "train.queued", f"{job_id} {file.filename} {len(content)}B")
    return {"job_id": job_id, "status": "queued"}


def _job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    return job or db.get_job(job_id) or {}


@app.get("/api/ml/status/{job_id}")
def status(job_id: str, cid: str = Auth) -> JSONResponse:
    job = _job(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id.")
    return JSONResponse(_sanitize({k: v for k, v in job.items() if k != "result"}))


@app.get("/api/ml/result/{job_id}")
def result(job_id: str, cid: str = Auth) -> JSONResponse:
    job = _job(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id.")
    if job.get("status") != "succeeded":
        raise HTTPException(409, f"Job is {job.get('status')}: {job.get('message')}")
    return JSONResponse(_sanitize(job["result"]))


@app.get("/api/ml/jobs")
def jobs(limit: int = 50, cid: str = Auth) -> list[dict[str, Any]]:
    return db.list_jobs(max(1, min(200, limit)))


@app.get("/api/ml/runs")
def runs(cid: str = Auth) -> list[dict[str, Any]]:
    return _load_runs()


@app.get("/api/ml/monitoring")
def monitoring_snapshot(cid: str = Auth) -> dict[str, Any]:
    snap = monitoring.snapshot()
    snap["inference"] = db.prediction_stats()
    snap["modelVersion"] = registry.current_version()
    snap["recentPredictions"] = db.list_predictions(15)
    return _sanitize(snap)


@app.get("/api/ml/retraining")
def retraining(cid: str = Auth) -> dict[str, Any]:
    return retrain.status()


@app.post("/api/ml/retraining/resume")
def retraining_resume(cid: str = Auth) -> dict[str, Any]:
    return retrain.resume()


@app.post("/api/ml/retraining/run")
def retraining_run(cid: str = Auth) -> dict[str, Any]:
    if not retrain.DATASET_SNAPSHOT.exists():
        raise HTTPException(409, "No dataset snapshot yet. Upload and train once first.")
    security.limit_training(cid)
    job_id = _start_job(retrain.DATASET_SNAPSHOT.read_bytes(), "manual-retrain")
    retrain._save(lastRunAt=time.time(), lastJobId=job_id, lastStatus="started")
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/ml/metrics")
def metrics(cid: str = Auth) -> dict[str, Any]:
    p = ARTIFACTS / "model.json"
    if not p.exists():
        raise HTTPException(404, "No trained model artifact yet.")
    return json.loads(p.read_text())["metrics"]


PREDICT_DOC = {
    "features": {f: 0.0 for f in FEATURES},
    "targetDate": "2026-01-31",
}


def _feature_vector(body: dict[str, Any]) -> tuple[list[float], dict[str, float]]:
    """Accepts either a flat/nested feature map or a raw `history` array (>=14 values)."""
    feats = body.get("features") if isinstance(body.get("features"), dict) else body
    if isinstance(body.get("history"), list) and len(body["history"]) >= 14:
        import numpy as np
        hist = np.asarray([float(v) for v in body["history"]], dtype="float64")
        when = pd.Timestamp(body.get("targetDate") or pd.Timestamp.utcnow().date())
        from preprocessing import build_forecast_features
        row = build_forecast_features(hist, when)[0]
        values = {f: float(row[i]) for i, f in enumerate(FEATURES)}
        return [values[f] for f in FEATURES], values
    missing = [f for f in FEATURES if f not in (feats or {})]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}. Send all {len(FEATURES)} "
                                 f"features, or send a `history` array of >= 14 past values.")
    try:
        values = {f: float(feats[f]) for f in FEATURES}
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"All feature values must be numeric: {exc}") from exc
    return [values[f] for f in FEATURES], values


def _predict(body: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    version = registry.current_version()
    target_date = str(body.get("targetDate") or "") or None
    try:
        x, values = _feature_vector(body)
        model, meta = registry.load_current()
        delta = float(model.predict([x])[0])
        predicted = delta + values["lag_1"]
        latency = (time.perf_counter() - t0) * 1000
        db.add_prediction(meta.get("version"), meta.get("modelName"), latency, True,
                          target_date, predicted)
        drift.record(meta.get("version"), values)
        return {
            "predicted": predicted,
            "model": meta.get("modelName"),
            "modelVersion": meta.get("version"),
            "targetDate": target_date,
            "latencyMs": round(latency, 2),
            "predictedAt": time.time(),
        }
    except FileNotFoundError as exc:
        db.add_prediction(version, None, (time.perf_counter() - t0) * 1000, False,
                          target_date, None, str(exc))
        raise HTTPException(404, str(exc)) from exc
    except HTTPException as exc:
        db.add_prediction(version, None, (time.perf_counter() - t0) * 1000, False,
                          target_date, None, str(exc.detail))
        raise
    except Exception as exc:
        db.add_prediction(version, None, (time.perf_counter() - t0) * 1000, False,
                          target_date, None, str(exc))
        monitoring.event("error", "predict.failed", str(exc))
        raise HTTPException(500, f"Prediction failed: {exc}") from exc


@app.post("/api/ml/predict")
def predict(body: dict[str, Any], cid: str = Auth) -> dict[str, Any]:
    return _predict(body)


@app.post("/predict")
def predict_alias(body: dict[str, Any], cid: str = Auth) -> dict[str, Any]:
    return _predict(body)


@app.get("/health")
def health_alias() -> dict[str, Any]:
    return health()


@app.get("/model/info")
@app.get("/api/ml/model/info")
def model_info(cid: str = Auth) -> dict[str, Any]:
    meta = registry.get_metadata()
    if not meta:
        raise HTTPException(404, "No model has been registered yet. Train a model first.")
    return _sanitize({**meta, "deployed": True, "featureCount": len(meta.get("features", []))})


@app.get("/api/ml/registry")
def model_registry(cid: str = Auth) -> dict[str, Any]:
    idx = registry.list_versions()
    return _sanitize({**idx, "currentMetadata": registry.get_metadata()})


@app.post("/api/ml/registry/activate/{version}")
def registry_activate(version: int, cid: str = Auth) -> dict[str, Any]:
    try:
        out = registry.activate(version)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    monitoring.event("info", "registry.activated", f"v{version}")
    return _sanitize({**out, "currentMetadata": registry.get_metadata()})


@app.get("/api/ml/drift")
def drift_status(cid: str = Auth) -> dict[str, Any]:
    return _sanitize(drift.status(registry.current_version()))


@app.get("/api/ml/performance")
def performance_report(cid: str = Auth) -> dict[str, Any]:
    return _sanitize(performance.report())


@app.post("/api/ml/actuals")
def post_actuals(body: dict[str, Any], cid: str = Auth) -> dict[str, Any]:
    records = body.get("records") if isinstance(body, dict) else None
    if not isinstance(records, list):
        raise HTTPException(422, 'Send {"records": [{"date": "2026-01-31", "actual": 1234.5}]}')
    try:
        return _sanitize(performance.add_actuals(records))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/ml/predictions")
def recent_predictions(limit: int = 50, cid: str = Auth) -> list[dict[str, Any]]:
    return _sanitize(db.list_predictions(max(1, min(200, limit))))


@app.get("/api/ml/endpoints")
def api_docs() -> dict[str, Any]:
    """Machine-readable API documentation rendered by the UI (OpenAPI lives at /docs)."""
    return {
        "openapi": "/docs",
        "baseUrl": "",
        "auth": "x-api-key header when ML_API_KEYS is set",
        "endpoints": [
            {"method": "GET", "path": "/health", "summary": "Liveness, deployed model, security & retraining config",
             "response": {"status": "ok", "registered_model": "Ridge Regression", "registered_version": 1}},
            {"method": "GET", "path": "/model/info", "summary": "Deployed model metadata (name, timestamp, features, dataset id, metrics)",
             "response": {"version": 1, "modelName": "Ridge Regression", "features": FEATURES[:3] + ["..."]}},
            {"method": "POST", "path": "/predict", "summary": "Real inference from the persisted registered model",
             "request": PREDICT_DOC,
             "altRequest": {"history": [1200.0, "... >= 14 past values"], "targetDate": "2026-01-31"},
             "response": {"predicted": 1234.56, "model": "Ridge Regression", "modelVersion": 1, "latencyMs": 3.1}},
            {"method": "POST", "path": "/api/ml/train", "summary": "Multipart CSV upload; starts a background training job",
             "request": {"file": "<electricity.csv>", "horizon": 30}, "response": {"job_id": "run-abc123", "status": "queued"}},
            {"method": "GET", "path": "/api/ml/status/{job_id}", "summary": "Job status + live progress",
             "response": {"status": "running", "progress": 45, "message": "Training Random Forest"}},
            {"method": "GET", "path": "/api/ml/result/{job_id}", "summary": "Full comparison, metrics, importances, forecast"},
            {"method": "GET", "path": "/api/ml/runs", "summary": "Experiment history (MLflow-mirrored run records)"},
            {"method": "GET", "path": "/api/ml/registry", "summary": "All persisted model versions + deployed pointer"},
            {"method": "POST", "path": "/api/ml/registry/activate/{version}", "summary": "Deploy a previous model version for inference"},
            {"method": "GET", "path": "/api/ml/monitoring", "summary": "Uptime, job counters, inference counters, latency, events"},
            {"method": "GET", "path": "/api/ml/drift", "summary": "PSI drift of live inference features vs the training reference"},
            {"method": "GET", "path": "/api/ml/performance", "summary": "Live MAE/MSE/RMSE/R2/MAPE vs the original test metrics"},
            {"method": "POST", "path": "/api/ml/actuals", "summary": "Post observed values so live performance can be computed",
             "request": {"records": [{"date": "2026-01-31", "actual": 1234.5}]}},
            {"method": "GET", "path": "/api/ml/predictions", "summary": "Recent inference log (latency, version, timestamp)"},
            {"method": "GET", "path": "/api/ml/jobs", "summary": "Training job history"},
            {"method": "GET", "path": "/api/ml/retraining", "summary": "Retraining scheduler state"},
            {"method": "POST", "path": "/api/ml/retraining/run", "summary": "Trigger a retraining run on the last dataset snapshot"},
            {"method": "POST", "path": "/api/ml/retraining/resume", "summary": "Clear a paused retraining scheduler"},
        ],
    }
