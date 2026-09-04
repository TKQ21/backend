"""Automated retraining scheduler.

Every RETRAIN_INTERVAL_HOURS the scheduler retrains on the last successfully
trained dataset (snapshotted to artifacts/last_dataset.csv).

Safety rules:
  * bounded work: exactly ONE retrain per tick
  * single-flight DB lease so two workers never train in parallel
  * paused state persisted in the DB after repeated failures; every entry point
    checks it and exits, and a paused job only runs one probe per tick
  * enable with RETRAIN_ENABLED=1
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

import db
import monitoring

ARTIFACTS = Path(__file__).parent / "artifacts"
DATASET_SNAPSHOT = ARTIFACTS / "last_dataset.csv"
ENABLED = os.getenv("RETRAIN_ENABLED", "0") == "1"
INTERVAL_HOURS = float(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))
MAX_FAILURES = 3
STATE_KEY = "retrain:state"


def _state() -> dict:
    return db.kv_get(STATE_KEY, {}) or {}


def _save(**kw) -> None:
    s = _state()
    s.update(kw)
    db.kv_set(STATE_KEY, s)


def snapshot_dataset(content: bytes) -> None:
    try:
        DATASET_SNAPSHOT.write_bytes(content)
        _save(datasetBytes=len(content), datasetAt=time.time())
    except Exception as exc:
        monitoring.event("error", "retrain.snapshot_failed", str(exc))


def status() -> dict:
    s = _state()
    return {
        "enabled": ENABLED,
        "intervalHours": INTERVAL_HOURS,
        "paused": bool(s.get("paused")),
        "pauseReason": s.get("pauseReason"),
        "failures": s.get("failures", 0),
        "lastRunAt": s.get("lastRunAt"),
        "lastJobId": s.get("lastJobId"),
        "lastStatus": s.get("lastStatus"),
        "nextRunAt": (s.get("lastRunAt") or 0) + INTERVAL_HOURS * 3600 if ENABLED else None,
        "datasetReady": DATASET_SNAPSHOT.exists(),
    }


def resume() -> dict:
    _save(paused=False, pauseReason=None, failures=0)
    monitoring.event("info", "retrain.resumed")
    return status()


def _tick(runner: Callable[[bytes, str], str]) -> None:
    s = _state()
    if not DATASET_SNAPSHOT.exists():
        return
    if s.get("paused") and not s.get("probeAllowed", True):
        return
    if not db.acquire_lease("retrain", ttl_seconds=int(INTERVAL_HOURS * 3600)):
        return
    try:
        content = DATASET_SNAPSHOT.read_bytes()
        job_id = runner(content, "scheduled")
        _save(lastRunAt=time.time(), lastJobId=job_id, lastStatus="started")
        monitoring.event("info", "retrain.started", job_id)
    except Exception as exc:
        fails = _state().get("failures", 0) + 1
        paused = fails >= MAX_FAILURES
        _save(failures=fails, paused=paused,
              pauseReason=str(exc)[:300] if paused else None, lastStatus="failed")
        monitoring.event("error", "retrain.failed", f"attempt {fails}: {exc}")
    finally:
        db.release_lease("retrain")


def start(runner: Callable[[bytes, str], str]) -> None:
    if not ENABLED:
        return

    def loop() -> None:
        time.sleep(30)  # cooldown after boot
        while True:
            try:
                _tick(runner)
            except Exception as exc:
                monitoring.event("error", "retrain.loop_error", str(exc))
            time.sleep(max(300, INTERVAL_HOURS * 3600))

    threading.Thread(target=loop, daemon=True).start()
    monitoring.event("info", "retrain.scheduler_started", f"every {INTERVAL_HOURS}h")
