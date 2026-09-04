"""Live model-performance monitoring.

Once real observed values are posted for dates the model already predicted,
the same metric functions used during training (models.evaluate) recompute
MAE / MSE / RMSE / R2 / MAPE on those matched pairs and compare them against
the test metrics recorded for the deployed model version.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

import db
import registry
from models import evaluate

MIN_PAIRS = 5


def _delta(live: float | None, ref: float | None) -> float | None:
    if live is None or ref is None:
        return None
    return round(live - ref, 6)


def report() -> dict[str, Any]:
    meta = registry.get_metadata()
    ref = ((meta or {}).get("metrics") or {}).get("test") or {}
    pairs = db.matched_predictions()
    out: dict[str, Any] = {
        "modelVersion": (meta or {}).get("version"),
        "modelName": (meta or {}).get("modelName"),
        "matchedPairs": len(pairs),
        "minPairs": MIN_PAIRS,
        "referenceTestMetrics": ref or None,
        "liveMetrics": None,
        "delta": None,
        "degraded": None,
        "computedAt": time.time(),
        "recent": pairs[-30:],
    }
    if len(pairs) < MIN_PAIRS:
        out["status"] = "awaiting-actuals"
        return out

    y_true = np.asarray([p["actual"] for p in pairs], dtype="float64")
    y_pred = np.asarray([p["predicted"] for p in pairs], dtype="float64")
    live = evaluate(y_true, y_pred)
    out["liveMetrics"] = live
    out["delta"] = {k: _delta(live.get(k), ref.get(k)) for k in ("mae", "mse", "rmse", "r2", "mape")}
    if ref.get("rmse") is not None and live.get("rmse") is not None:
        out["degraded"] = bool(live["rmse"] > ref["rmse"] * 1.25)
    out["status"] = "ok"
    return out


def add_actuals(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[str, float]] = []
    for r in records:
        d = str(r.get("date") or r.get("target_date") or "").strip()
        v = r.get("actual", r.get("value"))
        if not d or v is None:
            continue
        try:
            pairs.append((d, float(v)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        raise ValueError("No valid {date, actual} records were supplied.")
    db.upsert_actuals(pairs)
    return {"stored": len(pairs), **report()}
