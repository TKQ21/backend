"""Model zoo + metrics. Optional gradient-boosting libs degrade gracefully."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

MetricDict = dict[str, float | None]


def build_models(n_jobs: int = -1) -> tuple[list[tuple[str, str, Any]], list[dict[str, str]]]:
    """Returns (zoo, skipped). zoo = (display name, family, estimator).

    Optional gradient-boosting libraries that are not installed are reported in
    `skipped` so the UI can say "Skipped - dependency unavailable" instead of
    silently pretending the model never existed.
    """
    zoo: list[tuple[str, str, Any]] = [
        ("Linear Regression", "linear", LinearRegression()),
        ("Ridge Regression", "linear", Ridge(alpha=1.0)),
        ("Decision Tree", "tree", DecisionTreeRegressor(max_depth=8, min_samples_leaf=3, random_state=42)),
        ("Random Forest", "ensemble", RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=2, n_jobs=n_jobs, random_state=42)),
        ("KNN Regressor", "instance", KNeighborsRegressor(n_neighbors=5, n_jobs=n_jobs)),
    ]
    skipped: list[dict[str, str]] = []
    try:
        from xgboost import XGBRegressor
        zoo.append(("XGBoost", "boosting", XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", n_jobs=n_jobs, random_state=42)))
    except Exception as exc:  # pragma: no cover - optional dependency
        skipped.append({"name": "XGBoost", "family": "boosting", "package": "xgboost", "reason": str(exc)[:160]})
    try:
        from lightgbm import LGBMRegressor
        zoo.append(("LightGBM", "boosting", LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.9,
            n_jobs=n_jobs, random_state=42, verbose=-1)))
    except Exception as exc:  # pragma: no cover
        skipped.append({"name": "LightGBM", "family": "boosting", "package": "lightgbm", "reason": str(exc)[:160]})
    try:
        from catboost import CatBoostRegressor
        zoo.append(("CatBoost", "boosting", CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=42,
            allow_writing_files=False)))
    except Exception as exc:  # pragma: no cover
        skipped.append({"name": "CatBoost", "family": "boosting", "package": "catboost", "reason": str(exc)[:160]})
    return zoo, skipped


def iteration_info(family: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Boosting rounds / trees / neighbours actually configured — never called 'epochs'."""
    if family == "boosting":
        n = params.get("n_estimators") or params.get("iterations")
        return {"unit": "boosting rounds", "value": n} if n else None
    if family == "ensemble":
        n = params.get("n_estimators")
        return {"unit": "trees", "value": n} if n else None
    if family == "tree":
        return {"unit": "max depth", "value": params.get("max_depth")}
    if family == "instance":
        return {"unit": "neighbours", "value": params.get("n_neighbors")}
    return {"unit": "closed-form fit", "value": None}



def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> MetricDict:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if y_true.size == 0:
        return {"mae": None, "mse": None, "rmse": None, "r2": None, "mape": None}
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    mape = None
    if not np.any(np.abs(y_true) < 1e-9):
        mape = float(np.mean(np.abs(err / y_true)) * 100)
    return {
        "mae": float(np.mean(np.abs(err))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else None,
        "mape": mape,
    }


def importance(model: Any, n_features: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype="float64")
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_, dtype="float64")).ravel()
    return np.zeros(n_features)