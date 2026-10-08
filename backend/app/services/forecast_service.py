"""Forecast service: loads the trained demand models and produces
24/48/72-hour projections from the latest observed operational day.

Features are rebuilt exactly as in training (same function, same lags) so
training/inference preprocessing cannot drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "Bed-prediction-main" / "data" / "processed"
MODEL_DIR = ROOT / "models"

TARGETS = {"icu": "ICU_Demand", "emergency": "Emergency_Demand",
           "general_beds": "General_Demand"}
HORIZONS = {1: 24, 2: 48, 3: 72}

_cache: dict = {}


def _load_history() -> pd.DataFrame:
    df = pd.read_csv(RAW / "final_hospital_forecasting_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def _add_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mirror of training/train_demand_models.py::add_features (kept in sync)."""
    out = pd.DataFrame({"Date": df["Date"]})
    s = df[col]
    for lag in (1, 2, 3, 7, 14):
        out[f"{col}_lag{lag}"] = s.shift(lag)
    out[f"{col}_rollmean3"] = s.shift(1).rolling(3).mean()
    out[f"{col}_rollstd3"] = s.shift(1).rolling(3).std()
    out[f"{col}_rollmean7"] = s.shift(1).rolling(7).mean()
    for ex in ("Staff_Count", "Flu_Cases", "Bed_Occupancy", "discharges",
               "Admissions"):
        out[f"{ex}_t"] = df[ex]
    out["day_of_week"] = df["Date"].dt.dayofweek
    out["month"] = df["Date"].dt.month
    return out


def load_models() -> dict:
    if not _cache:
        for key in TARGETS:
            _cache[key] = joblib.load(MODEL_DIR / f"{key}_demand_model.pkl")
        _cache["history"] = _load_history()
        _cache["metrics"] = json.loads(
            (MODEL_DIR / "forecast_metrics.json").read_text())
    return _cache


def current_state_row() -> pd.Series:
    """Most recent observed day - the 'Now' point of every forecast."""
    return load_models()["history"].iloc[-1]


def forecast(horizon_hours: int = 72) -> dict:
    """Predict demand at +24h, +48h, +72h for each series.

    The origin row (latest observed day) supplies lag/exogenous features;
    horizon_days tells the model which lead time to predict.
    """
    if horizon_hours not in HORIZONS.values():
        raise ValueError("horizon_hours must be 24, 48 or 72")
    c = load_models()
    df = c["history"]
    origin = df.iloc[[-1]].copy()
    origin_date = origin["Date"].iloc[0]

    out = {"forecast_horizon": horizon_hours,
           "origin_date": origin_date.isoformat(),
           "timestamps": [], "series": {}}
    horizons = [h for h, hrs in sorted(HORIZONS.items()) if hrs <= horizon_hours]
    out["timestamps"] = [
        "now"] + [f"+{HORIZONS[h]}h" for h in horizons]
    out["timestamps_iso"] = [origin_date.isoformat()] + [
        (origin_date + pd.Timedelta(days=h)).isoformat() for h in horizons]

    for key, col in TARGETS.items():
        feats = _add_features(df, col)
        last = feats.iloc[-1].drop(labels=["Date"])
        preds = []
        for h in horizons:
            row = last.to_dict()
            row["horizon_days"] = h
            X = pd.DataFrame([{k: row[k] for k in c[key]["features"]}])
            model = c[key]["models"][h]
            preds.append(int(round(float(model.predict(X)[0]))))
        current = int(df[col].iloc[-1])
        out["series"][key] = [current] + preds
    # Percent-utilisation convenience series (existing charts show %/index).
    out["icu"] = out["series"]["icu"]
    out["emergency"] = out["series"]["emergency"]
    out["general"] = out["series"]["general_beds"]
    return out


def demand_for_horizon(horizon_hours: int = 48) -> dict:
    """Latest observed values + forecast at the chosen horizon (feed optimizer)."""
    f = forecast(horizon_hours)
    idx = {24: 1, 48: 2, 72: 3}[horizon_hours]
    return {
        "icu": f["series"]["icu"][idx],
        "emergency": f["series"]["emergency"][idx],
        "general": f["series"]["general_beds"][idx],
        "current": {
            "icu": f["series"]["icu"][0],
            "emergency": f["series"]["emergency"][0],
            "general": f["series"]["general_beds"][0],
        },
        "forecast": f,
    }
