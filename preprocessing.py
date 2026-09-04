"""Data cleaning + feature engineering for the electricity demand pipeline.

Memory-conscious: the raw frame is reduced to two columns (timestamp, target)
immediately, downcast to float32/datetime64, and every later step works in
place on that single frame. No full copies of a 300k-row dataset are kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "roll_mean_7", "roll_std_7", "roll_mean_14", "diff_1",
    "dow_sin", "dow_cos", "is_weekend", "month_sin", "month_cos", "doy_sin", "doy_cos",
]

TARGET_HINT = r"consumption|kwh|mwh|gwh|load|demand|usage|electricity|power|energy"
DATE_HINT = r"date|time|day|period|month|ds$"


@dataclass
class SchemaReport:
    dateColumn: str | None = None
    targetColumn: str | None = None
    candidateTargets: list[str] = field(default_factory=list)
    totalRows: int = 0
    usableRows: int = 0
    droppedMissing: int = 0
    droppedNonNumeric: int = 0
    droppedOutliers: int = 0
    duplicateDates: int = 0
    medianGapDays: float = float("nan")
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not np.isfinite(d["medianGapDays"]):
            d["medianGapDays"] = None
        return d


def detect_schema(df: pd.DataFrame) -> SchemaReport:
    rep = SchemaReport(totalRows=int(len(df)))
    if df.empty:
        rep.errors.append("CSV contains no data rows.")
        return rep

    head = df.head(2000)
    cols = list(df.columns)

    def parse_score(c: str) -> float:
        s = pd.to_datetime(head[c], errors="coerce", format="mixed")
        return float(s.notna().mean())

    named = [c for c in cols if pd.Series([c]).str.contains(DATE_HINT, case=False, regex=True).iat[0]]
    scored = sorted(((parse_score(c), c) for c in cols), reverse=True)
    if named and parse_score(named[0]) > 0.4:
        rep.dateColumn = named[0]
    elif scored and scored[0][0] > 0.4:
        rep.dateColumn = scored[0][1]

    for c in cols:
        if c == rep.dateColumn:
            continue
        if pd.to_numeric(head[c], errors="coerce").notna().mean() > 0.6:
            rep.candidateTargets.append(c)

    hinted = [c for c in rep.candidateTargets
              if pd.Series([c]).str.contains(TARGET_HINT, case=False, regex=True).iat[0]]
    rep.targetColumn = hinted[0] if hinted else (rep.candidateTargets[0] if rep.candidateTargets else None)

    if not rep.dateColumn:
        rep.errors.append("No parseable date/timestamp column found.")
    if not rep.targetColumn:
        rep.errors.append("No numeric consumption/target column found.")
    return rep


def clean(df: pd.DataFrame, rep: SchemaReport) -> pd.DataFrame:
    """Return a 2-column frame [date, value], cleaned in place."""
    d = df[[rep.dateColumn, rep.targetColumn]].copy()
    del df
    d.columns = ["date", "value"]

    before = len(d)
    d = d[d["date"].notna() & (d["date"].astype(str).str.len() > 0)]
    d = d[d["value"].notna() & (d["value"].astype(str).str.len() > 0)]
    rep.droppedMissing = int(before - len(d))

    before = len(d)
    d["date"] = pd.to_datetime(d["date"], errors="coerce", format="mixed")
    d["value"] = pd.to_numeric(d["value"], errors="coerce", downcast="float")
    d = d.dropna()
    rep.droppedNonNumeric = int(before - len(d))

    d = d.sort_values("date", kind="mergesort")

    # duplicate timestamps -> mean (vectorised groupby, no python loop)
    n_before = len(d)
    d = d.groupby("date", as_index=False, sort=True)["value"].mean()
    rep.duplicateDates = int(n_before - len(d))
    rep.usableRows = int(len(d))  # recorded early so a later failure still reports real counts


    # robust winsorising via MAD (copy -> guaranteed writable, pandas-backed views are read-only)
    v = d["value"].to_numpy()
    v = np.asarray(v, dtype="float64").copy()
    med = float(np.median(v)) if v.size else 0.0
    mad = float(np.median(np.abs(v - med))) if v.size else 0.0
    if mad > 0:
        hi = med + 8 * 1.4826 * mad
        lo = max(0.0, med - 8 * 1.4826 * mad)
        mask = (v > hi) | (v < lo)
        rep.droppedOutliers = int(mask.sum())
        np.clip(v, lo, hi, out=v)
        d["value"] = v

    gaps = d["date"].diff().dt.total_seconds().to_numpy()[1:] / 86400.0
    rep.medianGapDays = float(np.median(gaps)) if gaps.size else float("nan")
    rep.usableRows = int(len(d))


    if rep.usableRows < 60:
        rep.warnings.append(
            f"Only {rep.usableRows} usable observations — metrics on a test split this small are "
            "unreliable. 200+ rows recommended."
        )
    if np.isfinite(rep.medianGapDays) and abs(rep.medianGapDays - 1) > 0.01:
        rep.warnings.append(
            f"Median gap between observations is {rep.medianGapDays:g} days (not daily) — "
            "lag features are index-based, not calendar-based."
        )
    return d.reset_index(drop=True)


def make_features(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Vectorised lag / rolling / calendar features. Returns (X, y, dates)."""
    s = d["value"].astype("float32")
    f = pd.DataFrame(index=d.index)
    for lag in (1, 2, 3, 7, 14):
        f[f"lag_{lag}"] = s.shift(lag)
    prev = s.shift(1)
    f["roll_mean_7"] = prev.rolling(7).mean()
    f["roll_std_7"] = prev.rolling(7).std(ddof=0)
    f["roll_mean_14"] = prev.rolling(14).mean()
    f["diff_1"] = f["lag_1"] - f["lag_2"]

    dt = d["date"].dt
    dow = dt.dayofweek.to_numpy()
    # pandas dayofweek: Monday=0 .. Sunday=6 ; align with JS getDay() (Sunday=0)
    dow_js = (dow + 1) % 7
    f["dow_sin"] = np.sin(2 * np.pi * dow_js / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow_js / 7)
    f["is_weekend"] = np.isin(dow_js, (0, 6)).astype("float32")
    month = dt.month.to_numpy()
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    doy = dt.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    f = f[FEATURES]
    valid = f.notna().all(axis=1)
    X = f.loc[valid].to_numpy(dtype="float32", copy=False)
    y = s.loc[valid].to_numpy(dtype="float32", copy=False)
    dates = d.loc[valid, "date"]
    return X, y, dates


def build_forecast_features(history: np.ndarray, when: pd.Timestamp) -> np.ndarray:
    """Feature row for one recursive forecast step (history = past values)."""
    w7 = history[-7:]
    w14 = history[-14:]
    dow_js = (when.dayofweek + 1) % 7
    return np.array([[
        history[-1], history[-2], history[-3], history[-7], history[-14],
        w7.mean(), w7.std(), w14.mean(), history[-1] - history[-2],
        np.sin(2 * np.pi * dow_js / 7), np.cos(2 * np.pi * dow_js / 7),
        1.0 if dow_js in (0, 6) else 0.0,
        np.sin(2 * np.pi * when.month / 12), np.cos(2 * np.pi * when.month / 12),
        np.sin(2 * np.pi * when.dayofyear / 365.25), np.cos(2 * np.pi * when.dayofyear / 365.25),
    ]], dtype="float32")