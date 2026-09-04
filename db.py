"""SQLite persistence for jobs, runs, events and the retraining lease.

Single file DB at backend/artifacts/mlops.db (override with ML_DB_PATH).
Everything is plain sqlite3 - no ORM, no external service.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
DB_PATH = Path(os.getenv("ML_DB_PATH", ARTIFACTS / "mlops.db"))

_LOCK = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  progress INTEGER DEFAULT 0,
  message TEXT,
  file_name TEXT,
  source TEXT DEFAULT 'upload',
  error TEXT,
  schema_json TEXT,
  result_json TEXT,
  created_at REAL,
  updated_at REAL
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  experiment TEXT,
  finished_at TEXT,
  best_model TEXT,
  registered_version INTEGER,
  dataset_json TEXT,
  metrics_json TEXT,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL,
  level TEXT,
  event TEXT,
  detail TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL,
  model_version INTEGER,
  model_name TEXT,
  latency_ms REAL,
  ok INTEGER,
  target_date TEXT,
  predicted REAL,
  error TEXT
);
CREATE TABLE IF NOT EXISTS actuals (
  target_date TEXT PRIMARY KEY,
  actual REAL,
  ts REAL
);
CREATE TABLE IF NOT EXISTS kv (
  k TEXT PRIMARY KEY,
  v TEXT
);
"""


def init() -> None:
    with _LOCK, _conn() as c:
        c.executescript(SCHEMA)


# ---------------------------------------------------------------- jobs
def upsert_job(job_id: str, **fields: Any) -> None:
    cols = {
        "status": fields.get("status"),
        "progress": fields.get("progress"),
        "message": fields.get("message"),
        "file_name": fields.get("fileName"),
        "source": fields.get("source"),
        "error": fields.get("error"),
        "schema_json": json.dumps(fields["schema"], default=str) if "schema" in fields else None,
        "result_json": json.dumps(fields["result"], default=str) if "result" in fields else None,
    }
    cols = {k: v for k, v in cols.items() if v is not None}
    now = time.time()
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO jobs(job_id, status, created_at, updated_at) VALUES(?,?,?,?)",
            (job_id, cols.get("status", "queued"), now, now),
        )
        if cols:
            sets = ", ".join(f"{k}=?" for k in cols)
            c.execute(
                f"UPDATE jobs SET {sets}, updated_at=? WHERE job_id=?",
                (*cols.values(), now, job_id),
            )


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK, _conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    out: dict[str, Any] = {
        "status": d["status"],
        "progress": d["progress"] or 0,
        "message": d["message"],
        "fileName": d["file_name"],
        "source": d["source"],
        "createdAt": d["created_at"],
    }
    if d["error"]:
        out["error"] = d["error"]
    if d["schema_json"]:
        out["schema"] = json.loads(d["schema_json"])
    if d["result_json"]:
        out["result"] = json.loads(d["result_json"])
    return out


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT job_id, status, progress, message, file_name, source, created_at, updated_at"
            " FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "jobId": r["job_id"],
            "status": r["status"],
            "progress": r["progress"],
            "message": r["message"],
            "fileName": r["file_name"],
            "source": r["source"],
            "createdAt": r["created_at"],
            "updatedAt": r["updated_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------- runs
def add_run(run: dict[str, Any]) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO runs(run_id, experiment, finished_at, best_model,"
            " registered_version, dataset_json, metrics_json, created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (
                run["runId"],
                run.get("experiment"),
                str(run.get("finishedAt")),
                run.get("bestModel"),
                run.get("registeredVersion"),
                json.dumps(run.get("dataset", {}), default=str),
                json.dumps(run.get("metrics", {}), default=str),
                time.time(),
            ),
        )


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "runId": r["run_id"],
            "experiment": r["experiment"],
            "finishedAt": r["finished_at"],
            "bestModel": r["best_model"],
            "registeredVersion": r["registered_version"],
            "dataset": json.loads(r["dataset_json"] or "{}"),
            "metrics": json.loads(r["metrics_json"] or "{}"),
        }
        for r in rows
    ]


# ---------------------------------------------------------------- events
def log_event(level: str, event: str, detail: str = "") -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO events(ts, level, event, detail) VALUES(?,?,?,?)",
            (time.time(), level, event, detail[:2000]),
        )
        c.execute(
            "DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 500)"
        )


def list_events(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT ts, level, event, detail FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- kv / lease
def kv_get(key: str, default: Any = None) -> Any:
    with _LOCK, _conn() as c:
        r = c.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
    return json.loads(r["v"]) if r else default


def kv_set(key: str, value: Any) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO kv(k, v) VALUES(?, ?)", (key, json.dumps(value, default=str))
        )


def acquire_lease(name: str, ttl_seconds: int) -> bool:
    """Single-flight lock: returns True only if this caller owns the lease."""
    now = time.time()
    with _LOCK, _conn() as c:
        r = c.execute("SELECT v FROM kv WHERE k=?", (f"lease:{name}",)).fetchone()
        if r:
            try:
                if float(json.loads(r["v"]).get("expires", 0)) > now:
                    return False
            except Exception:
                pass
        c.execute(
            "INSERT OR REPLACE INTO kv(k, v) VALUES(?, ?)",
            (f"lease:{name}", json.dumps({"expires": now + ttl_seconds})),
        )
    return True


def release_lease(name: str) -> None:
    with _LOCK, _conn() as c:
        c.execute("DELETE FROM kv WHERE k=?", (f"lease:{name}",))


# ---------------------------------------------------------------- inference
def add_prediction(model_version: int | None, model_name: str | None, latency_ms: float,
                   ok: bool, target_date: str | None, predicted: float | None,
                   error: str | None = None) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO predictions(ts, model_version, model_name, latency_ms, ok, target_date,"
            " predicted, error) VALUES(?,?,?,?,?,?,?,?)",
            (time.time(), model_version, model_name, latency_ms, 1 if ok else 0,
             target_date, predicted, (error or "")[:500] or None),
        )
        c.execute("DELETE FROM predictions WHERE id NOT IN"
                  " (SELECT id FROM predictions ORDER BY id DESC LIMIT 2000)")


def prediction_stats() -> dict[str, Any]:
    with _LOCK, _conn() as c:
        r = c.execute(
            "SELECT COUNT(*) n, SUM(ok) ok, AVG(latency_ms) avg_ms,"
            " MIN(latency_ms) min_ms, MAX(latency_ms) max_ms, MAX(ts) last_ts FROM predictions"
        ).fetchone()
        p50 = c.execute(
            "SELECT latency_ms FROM predictions WHERE ok=1 ORDER BY latency_ms"
            " LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM predictions WHERE ok=1)"
        ).fetchone()
    n = r["n"] or 0
    ok = r["ok"] or 0
    return {
        "predictionCount": n,
        "successfulRequests": ok,
        "failedRequests": n - ok,
        "avgLatencyMs": round(r["avg_ms"], 2) if r["avg_ms"] is not None else None,
        "minLatencyMs": round(r["min_ms"], 2) if r["min_ms"] is not None else None,
        "maxLatencyMs": round(r["max_ms"], 2) if r["max_ms"] is not None else None,
        "p50LatencyMs": round(p50["latency_ms"], 2) if p50 else None,
        "lastPredictionAt": r["last_ts"],
    }


def list_predictions(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT ts, model_version, model_name, latency_ms, ok, target_date, predicted, error"
            " FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [
        {"at": r["ts"], "modelVersion": r["model_version"], "model": r["model_name"],
         "latencyMs": round(r["latency_ms"], 2) if r["latency_ms"] is not None else None,
         "ok": bool(r["ok"]), "targetDate": r["target_date"],
         "predicted": r["predicted"], "error": r["error"]}
        for r in rows
    ]


def upsert_actuals(pairs: list[tuple[str, float]]) -> int:
    now = time.time()
    with _LOCK, _conn() as c:
        c.executemany("INSERT OR REPLACE INTO actuals(target_date, actual, ts) VALUES(?,?,?)",
                      [(d, float(v), now) for d, v in pairs])
    return len(pairs)


def matched_predictions() -> list[dict[str, Any]]:
    """Predictions that now have a real observed value for the same target date."""
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT p.target_date d, p.predicted pred, a.actual act, p.model_version v,"
            " p.model_name m, p.ts ts FROM predictions p JOIN actuals a"
            " ON a.target_date = p.target_date WHERE p.ok=1 AND p.predicted IS NOT NULL"
            " ORDER BY p.target_date"
        ).fetchall()
    return [{"date": r["d"], "predicted": r["pred"], "actual": r["act"],
             "modelVersion": r["v"], "model": r["m"], "at": r["ts"]} for r in rows]


def counts() -> dict[str, int]:
    with _LOCK, _conn() as c:
        q = lambda s, *a: c.execute(s, a).fetchone()[0]  # noqa: E731
        return {
            "jobs": q("SELECT COUNT(*) FROM jobs"),
            "succeeded": q("SELECT COUNT(*) FROM jobs WHERE status='succeeded'"),
            "failed": q("SELECT COUNT(*) FROM jobs WHERE status='failed'"),
            "running": q("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"),
            "runs": q("SELECT COUNT(*) FROM runs"),
            "errors": q("SELECT COUNT(*) FROM events WHERE level='error'"),
        }


init()
