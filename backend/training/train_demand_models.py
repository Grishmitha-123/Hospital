"""Train demand forecasting models: ICU / Emergency / General beds.

Data source: Bed-prediction dataset
(final_hospital_forecasting_dataset.csv, 208 consecutive daily
hospital-operations records). SYNTHETIC/simulated operational data.

Design
------
* Direct multi-horizon regression: one model per demand series, trained to
  predict demand h days ahead (h in {1,2,3} == 24/48/72 hours) from features
  known at forecast time.
* Features: target lags (1,2,3,7,14), rolling mean/std, calendar features and
  known-at-forecast operational signals (staff, flu). No future values used.
* Chronological split: earliest 70% train, next 15% validation, latest 15% test.
* Candidate models compared on validation MAE; best is refit on train+val.

Run:  python training/train_demand_models.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Bed-prediction-main" / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGETS = {
    "icu": "ICU_Demand",
    "emergency": "Emergency_Demand",
    "general_beds": "General_Demand",
}
HORIZONS = [1, 2, 3]  # days ahead == 24/48/72 hours
HORIZON_NAMES = {1: "24h", 2: "48h", 3: "72h"}


def load_daily() -> pd.DataFrame:
    df = pd.read_csv(RAW / "final_hospital_forecasting_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    print("daily rows:", len(df), "| range:", df["Date"].min().date(),
          "->", df["Date"].max().date())
    print("missing values:", int(df.isna().sum().sum()))
    for col in TARGETS.values():
        s = df[col]
        assert (s.diff().dropna() != 0).any(), f"{col} is constant"
        print(f"  {col}: min={s.min()} max={s.max()} mean={s.mean():.1f}")
    return df


def add_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Temporal features computed strictly from information at time t."""
    out = pd.DataFrame({"Date": df["Date"]})
    s = df[col]
    for lag in (1, 2, 3, 7, 14):
        out[f"{col}_lag{lag}"] = s.shift(lag)
    out[f"{col}_rollmean3"] = s.shift(1).rolling(3).mean()
    out[f"{col}_rollstd3"] = s.shift(1).rolling(3).std()
    out[f"{col}_rollmean7"] = s.shift(1).rolling(7).mean()
    # Exogenous signals known at forecast time: today's values (day t) are
    # observed before the forecast is issued, so they are used un-lagged.
    # Only targets are lagged - never future values.
    for ex in ("Staff_Count", "Flu_Cases", "Bed_Occupancy", "discharges",
               "Admissions"):
        out[f"{ex}_t"] = df[ex]
    out["day_of_week"] = df["Date"].dt.dayofweek
    out["month"] = df["Date"].dt.month
    # Calendar features for the *forecast target* date are derived from the
    # horizon, not from future outcomes -> safe.
    return out


def build_supervised(df: pd.DataFrame, col: str, horizon: int | None = None):
    """Supervised table. If horizon is None all horizons are stacked (used for
    algorithm selection); otherwise only one horizon's rows are returned."""
    feats = add_features(df, col)
    rows = []
    for h in (HORIZONS if horizon is None else [horizon]):
        X = feats.copy()
        X["horizon_days"] = h
        X["y"] = df[col].shift(-h)
        rows.append(X)
    all_rows = pd.concat(rows, ignore_index=True).dropna()
    y = all_rows.pop("y")
    return all_rows, y


def metrics(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def chronological_split(dates: pd.Series):
    """Earliest 70% -> train, next 15% -> validation, latest 15% -> test.
    Splits are by unique forecast-origin date so no future row leaks into
    earlier splits.
    """
    uniq = np.sort(dates.unique())
    n = len(uniq)
    c1, c2 = int(n * 0.70), int(n * 0.85)
    tr = set(uniq[:c1])
    va = set(uniq[c1:c2])
    te = set(uniq[c2:])
    return (dates.isin(tr).values, dates.isin(va).values, dates.isin(te).values)


def train_one(df: pd.DataFrame, key: str, col: str):
    X, y = build_supervised(df, col)
    dates = X["Date"]
    X = X.drop(columns=["Date"])
    num_cols = [c for c in X.columns]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
    ])

    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=4, n_jobs=-1, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=300, l2_regularization=1.0,
            random_state=42),
    }

    m_tr, m_va, m_te = chronological_split(dates)
    print(f"{key}: train={m_tr.sum()} val={m_va.sum()} test={m_te.sum()}")

    results, fitted = {}, {}
    for name, est in candidates.items():
        pipe = Pipeline([("pre", pre), ("reg", est)])
        pipe.fit(X[m_tr], y[m_tr])
        val_m = metrics(y[m_va], pipe.predict(X[m_va]))
        results[name] = {"validation": val_m}
        fitted[name] = pipe
        print(" ", name, val_m)

    best_name = min(results, key=lambda n: results[n]["validation"]["mae"])

    # One final model per horizon (direct multi-horizon strategy) so 24h, 48h
    # and 72h projections genuinely differ; algorithm chosen once per target.
    models_by_h, per_h = {}, {}
    for h in HORIZONS:
        Xh, yh = build_supervised(df, col, horizon=h)
        dh = Xh.pop("Date")
        h_tr, h_va, h_te = chronological_split(dh)
        model = Pipeline([
            ("pre", ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), list(Xh.columns))])),
            ("reg", clone(candidates[best_name])),
        ])
        model.fit(Xh[h_tr | h_va], yh[h_tr | h_va])
        per_h[HORIZON_NAMES[h]] = metrics(yh[h_te], model.predict(Xh[h_te]))
        models_by_h[h] = model
        print(f"  final {key} h={h}:", per_h[HORIZON_NAMES[h]])

    # Aggregate test metric across horizons (comparable to validation MAE).
    agg = {m: round(float(np.mean([per_h[k][m] for k in per_h])), 3)
           for m in ("mae", "rmse", "r2")}
    result = {
        "algorithm": best_name,
        "target_column": col,
        "selection_metric": "validation MAE (stacked horizons)",
        "validation": results[best_name]["validation"],
        "test": agg,
        "test_by_horizon": per_h,
        "all_candidates": results,
        "n_rows": int(len(X)),
        "split": "chronological 70/15/15 by forecast-origin date",
    }
    joblib.dump({"models": models_by_h, "features": list(X.columns)},
                MODEL_DIR / f"{key}_demand_model.pkl")
    return result


def main():
    df = load_daily()
    all_metrics = {}
    for key, col in TARGETS.items():
        print("=" * 60)
        all_metrics[key] = train_one(df, key, col)
        print(json.dumps(all_metrics[key], indent=2))

    (MODEL_DIR / "forecast_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    meta = {
        "models": list(TARGETS),
        "horizons_hours": [24, 48, 72],
        "source": "final_hospital_forecasting_dataset.csv (daily, synthetic ops data)",
        "features": "target lags 1/2/3/7/14, rolling mean/std, day_of_week, month, "
                    "staff/flu/occupancy/discharges lagged by 1 day, horizon_days",
        "split": "chronological 70/15/15 (no shuffling across time)",
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    (MODEL_DIR / "forecast_metadata.json").write_text(json.dumps(meta, indent=2))
    print("saved forecast artifacts to", MODEL_DIR)


if __name__ == "__main__":
    sys.exit(main())
