"""Real data-drift detection.

At registration time the training feature matrix is summarised into a reference
distribution (per feature: mean, std, and decile bin edges). Every inference
request appends its real feature vector to a bounded buffer; the drift endpoint
compares the buffered live distribution against the reference using the
Population Stability Index (PSI) plus a mean z-shift.

Thresholds (industry-standard PSI reading):
    PSI < 0.10  -> stable
    0.10-0.25   -> moderate drift
    > 0.25      -> significant drift

Nothing is invented: when fewer than MIN_SAMPLES predictions have been made the
status is "insufficient-data" and no numbers are reported.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

import db
from registry import MODELS_DIR

MIN_SAMPLES = 20
BUFFER_MAX = 500
BUFFER_KEY = "drift:buffer"
_LOCK = threading.Lock()


def _ref_path(version: int) -> Path:
    return MODELS_DIR / f"v{version}" / "reference.json"


def save_reference(version: int, X: np.ndarray, features: list[str]) -> None:
    X = np.asarray(X, dtype="float64")
    ref: dict[str, Any] = {
        "version": version,
        "rows": int(X.shape[0]),
        "createdAt": time.time(),
        "features": {},
    }
    qs = np.linspace(0, 100, 11)
    for i, f in enumerate(features):
        col = X[:, i]
        edges = np.unique(np.percentile(col, qs))
        counts, _ = np.histogram(col, bins=edges) if edges.size > 1 else (np.array([col.size]), None)
        ref["features"][f] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "edges": [float(e) for e in edges],
            "counts": [int(c) for c in np.atleast_1d(counts)],
        }
    _ref_path(version).write_text(json.dumps(ref))


def load_reference(version: int) -> dict[str, Any] | None:
    p = _ref_path(version)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def record(version: int | None, values: dict[str, float]) -> None:
    """Append one real inference feature vector to the rolling buffer."""
    with _LOCK:
        buf = db.kv_get(BUFFER_KEY, []) or []
        buf.append({"v": version, "t": time.time(), "x": values})
        db.kv_set(BUFFER_KEY, buf[-BUFFER_MAX:])


def reset() -> None:
    db.kv_set(BUFFER_KEY, [])


def _psi(ref_counts: list[int], edges: list[float], live: np.ndarray) -> float:
    ref = np.asarray(ref_counts, dtype="float64")
    if ref.sum() <= 0 or len(edges) < 2:
        return 0.0
    live_counts, _ = np.histogram(live, bins=np.asarray(edges, dtype="float64"))
    # values outside the reference range fall into the nearest edge bucket
    below = int(np.sum(live < edges[0]))
    above = int(np.sum(live > edges[-1]))
    live_counts = live_counts.astype("float64")
    live_counts[0] += below
    live_counts[-1] += above
    p = ref / ref.sum()
    q = live_counts / max(1.0, live_counts.sum())
    eps = 1e-6
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def status(version: int | None) -> dict[str, Any]:
    buf = [b for b in (db.kv_get(BUFFER_KEY, []) or []) if version is None or b.get("v") == version]
    ref = load_reference(version) if version is not None else None
    out: dict[str, Any] = {
        "modelVersion": version,
        "referenceAvailable": bool(ref),
        "referenceRows": (ref or {}).get("rows"),
        "liveSamples": len(buf),
        "minSamples": MIN_SAMPLES,
        "method": "Population Stability Index (decile bins) + mean z-shift",
        "thresholds": {"moderate": 0.1, "significant": 0.25},
        "status": "insufficient-data",
        "driftScore": None,
        "features": [],
        "checkedAt": time.time(),
        "lastPredictionAt": buf[-1]["t"] if buf else None,
    }
    if not ref:
        out["status"] = "no-reference"
        return out
    if len(buf) < MIN_SAMPLES:
        return out

    rows: list[dict[str, Any]] = []
    for f, r in ref["features"].items():
        live = np.asarray([b["x"].get(f) for b in buf if b["x"].get(f) is not None], dtype="float64")
        if live.size < MIN_SAMPLES:
            continue
        psi = _psi(r["counts"], r["edges"], live)
        std = r["std"] if r["std"] > 1e-9 else 1.0
        z = float((float(np.mean(live)) - r["mean"]) / std)
        rows.append({
            "feature": f,
            "psi": round(psi, 4),
            "zShift": round(z, 4),
            "referenceMean": round(r["mean"], 4),
            "liveMean": round(float(np.mean(live)), 4),
            "drifted": psi >= 0.25,
            "status": "significant" if psi >= 0.25 else ("moderate" if psi >= 0.1 else "stable"),
        })
    if not rows:
        return out
    rows.sort(key=lambda r: -r["psi"])
    worst = rows[0]["psi"]
    out.update({
        "status": "significant" if worst >= 0.25 else ("moderate" if worst >= 0.1 else "stable"),
        "driftScore": round(worst, 4),
        "meanPsi": round(float(np.mean([r["psi"] for r in rows])), 4),
        "driftedFeatures": [r["feature"] for r in rows if r["psi"] >= 0.25],
        "features": rows,
    })
    return out
