"""The training pipeline: clean -> features -> chronological split -> compare -> select -> forecast."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from models import build_models, evaluate, importance, iteration_info
from preprocessing import FEATURES, SchemaReport, build_forecast_features, clean, make_features
from mlflow_tracking import log_comparison
import drift
import registry

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

Progress = Callable[[int, str], None]


def _fit_predict(est: Any, Xtr, ytr, Xs: list[np.ndarray]) -> list[np.ndarray]:
    """Models learn the delta vs lag_1 (stationary target); level is added back."""
    est.fit(Xtr, ytr - Xtr[:, 0])
    return [est.predict(X) + X[:, 0] for X in Xs]


def run_pipeline(
    df: pd.DataFrame,
    rep: SchemaReport,
    horizon: int = 30,
    run_id: str = "",
    progress: Progress = lambda p, m: None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    progress(10, "Cleaning data")
    d = clean(df, rep)
    if rep.errors:
        raise ValueError("; ".join(rep.errors))

    progress(20, "Engineering features")
    X, y, dates = make_features(d)
    n = len(X)
    if n < 30:
        raise ValueError(
            f"Not enough usable observations after feature engineering ({n}). Need at least 44 raw rows."
        )

    n_test = max(5, int(n * 0.15))
    n_val = max(5, int(n * 0.15))
    n_train = n - n_val - n_test
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    Xte, yte = X[n_train + n_val:], y[n_train + n_val:]

    zoo, skipped = build_models()
    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for i, (name, family, est) in enumerate(zoo):
        progress(25 + int(55 * i / max(1, len(zoo))), f"Training {name}")
        t0 = time.perf_counter()
        try:
            pv, pt = _fit_predict(est, Xtr, ytr, [Xva, Xte])
        except Exception as exc:  # a model that cannot converge on this dataset is skipped
            results.append({"name": name, "family": family, "status": "failed", "error": str(exc)[:200]})
            continue
        imp = importance(est, len(FEATURES))
        total = float(imp.sum())
        params = {k: v for k, v in est.get_params().items()
                  if isinstance(v, (int, float, str, bool)) or v is None}
        results.append({
            "name": name,
            "family": family,
            "status": "trained",
            "params": params,
            "iterations": iteration_info(family, params),
            "val": evaluate(yva, pv),
            "test": evaluate(yte, pt),
            "trainSeconds": time.perf_counter() - t0,
            "importance": sorted(
                [{"feature": f, "value": (float(imp[j]) / total if total > 0 else 0.0)}
                 for j, f in enumerate(FEATURES)],
                key=lambda r: -r["value"]),
            "testPred": [float(v) for v in pt],
        })
        fitted[name] = est

    ok = [r for r in results if "val" in r and r["val"]["rmse"] is not None]
    if not ok:
        raise ValueError("All models failed to train on this dataset.")

    progress(85, "Selecting best model")
    best = min(ok, key=lambda r: (r["val"]["rmse"], r["val"]["mae"]))
    best_est = fitted[best["name"]]
    baseline = evaluate(yte, Xte[:, 0])  # naive lag-1 persistence

    progress(90, "Forecasting")
    step_days = max(1, int(round(rep.medianGapDays if np.isfinite(rep.medianGapDays) else 1)))
    history = d["value"].to_numpy(dtype="float64").copy()
    last = pd.Timestamp(d["date"].iloc[-1])
    forecast = []
    for _ in range(horizon):
        last = last + pd.Timedelta(days=step_days)
        feat = build_forecast_features(history, last)
        p = max(0.0, float(best_est.predict(feat)[0] + feat[0, 0]))
        forecast.append({"date": last.strftime("%Y-%m-%d"), "predicted": round(p, 2)})
        history = np.append(history, p)

    # Chart downsampling: metrics above are computed on every test point; only the
    # plotted series is thinned so the browser never renders 100k points.
    MAX_CHART_POINTS = 600
    _stride = max(1, int(np.ceil(len(yte) / MAX_CHART_POINTS)))
    for r in results:
        if isinstance(r.get("testPred"), list):
            r["testPred"] = r["testPred"][::_stride]

    progress(95, "Logging to MLflow and saving artifacts")
    payload: dict[str, Any] = {
        "schema": rep.dict(),
        "splits": {"train": n_train, "val": n_val, "test": n_test},
        "results": results,
        "best": best,
        "baseline": baseline,
        "testDates": [t.strftime("%Y-%m-%d") for t in dates.iloc[n_train + n_val:]][::_stride],
        "testActual": [float(v) for v in yte][::_stride],
        "chartStride": _stride,
        "skipped": skipped,
        "forecast": forecast,
        "runId": run_id or f"run-{int(time.time())}",
        "finishedAt": pd.Timestamp.utcnow().isoformat(),
        "features": FEATURES,
    }

    tracking = log_comparison(payload, best_est)
    registered = tracking.get("version")
    payload["registeredVersion"] = registered
    payload["mlflow"] = tracking

    # --- model persistence: immutable versioned artifact + deployed pointer -----
    reg_meta = registry.register(best_est, {
        "modelName": best["name"],
        "family": best["family"],
        "trainedAt": payload["finishedAt"],
        "runId": payload["runId"],
        "datasetId": dataset_id,
        "datasetRows": rep.usableRows,
        "dateColumn": rep.dateColumn,
        "targetColumn": rep.targetColumn,
        "features": FEATURES,
        "params": best["params"],
        "metrics": {"validation": best["val"], "test": best["test"],
                    "naive_baseline_test": baseline},
        "mlflowVersion": registered,
        "mlflowRunId": tracking.get("runId"),
    })
    drift.save_reference(reg_meta["version"], Xtr, FEATURES)
    payload["modelVersion"] = reg_meta["version"]
    payload["registry"] = {k: reg_meta[k] for k in
                           ("version", "modelName", "trainedAt", "datasetId", "artifactPath")}

    joblib.dump(best_est, ARTIFACTS / "model.pkl")
    meta = {
        "schema_version": 1,
        "run_id": payload["runId"],
        "registered_version": registered,
        "created_at": payload["finishedAt"],
        "dataset": {
            "date_column": rep.dateColumn,
            "target_column": rep.targetColumn,
            "usable_rows": rep.usableRows,
            "median_gap_days": rep.dict()["medianGapDays"],
        },
        "preprocessing": {
            "steps": ["drop_missing", "coerce_numeric", "sort_by_date",
                      "merge_duplicate_timestamps", "winsorize_mad_8",
                      "lag_and_calendar_features", "delta_vs_lag1_target"],
            "features": FEATURES,
        },
        "model": {"name": best["name"], "family": best["family"], "params": best["params"]},
        "metrics": {"validation": best["val"], "test": best["test"], "naive_baseline_test": baseline},
        "split": payload["splits"],
    }
    (ARTIFACTS / "model.json").write_text(json.dumps(meta, indent=2, default=str))
    progress(100, "Done")
    return payload