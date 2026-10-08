"""Patient risk inference service.

Rebuilds the exact admission-hour feature row used in training, runs the
saved (calibrated) model, and derives contributing factors from the model's
feature importance plus cohort reference statistics - no fabricated reasons.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "hospital-deterioration-dataset-main" / "data"
MODEL_DIR = ROOT / "models"

_cache: dict = {}

# Human labels for model feature names (frontend-facing).
FEATURE_LABELS = {
    "age": "age", "comorbidity_index": "comorbidity burden",
    "heart_rate_adm": "heart rate", "respiratory_rate_adm": "respiratory rate",
    "spo2_pct_adm": "oxygen saturation (SpO2)", "temperature_c_adm": "temperature",
    "systolic_bp_adm": "systolic blood pressure",
    "diastolic_bp_adm": "diastolic blood pressure",
    "oxygen_flow_adm": "oxygen flow", "mobility_score_adm": "mobility score",
    "nurse_alert_adm": "nurse alert at admission",
    "wbc_count_adm": "white blood cell count", "lactate_adm": "lactate",
    "creatinine_adm": "creatinine", "crp_level_adm": "CRP",
    "hemoglobin_adm": "hemoglobin", "oxygen_support_adm": "oxygen support level",
    "gender": "gender", "admission_type": "admission type",
}
# Direction that raises risk (used to phrase factor sentences honestly).
HIGH_IS_BAD = {
    "heart_rate_adm": True, "respiratory_rate_adm": True, "temperature_c_adm": True,
    "lactate_adm": True, "wbc_count_adm": True, "crp_level_adm": True,
    "creatinine_adm": True, "age": True, "comorbidity_index": True,
    "oxygen_flow_adm": True, "oxygen_support_adm": True, "nurse_alert_adm": True,
    "spo2_pct_adm": False, "systolic_bp_adm": False, "diastolic_bp_adm": False,
    "hemoglobin_adm": False, "mobility_score_adm": False,
}


def _load() -> dict:
    if not _cache:
        bundle = joblib.load(MODEL_DIR / "patient_risk_model.pkl")
        _cache["model"] = bundle["model"]
        _cache["calibrator"] = bundle["calibrator"]
        _cache["meta"] = json.loads((MODEL_DIR / "model_metadata.json").read_text())
        _cache["features"] = json.loads((MODEL_DIR / "feature_columns.json").read_text())
        _cache["reference"] = json.loads((MODEL_DIR / "risk_reference.json").read_text())
        _cache["metrics"] = json.loads((MODEL_DIR / "metrics.json").read_text())
        _cache["frame"] = _build_frame()
    return _cache


def _build_frame() -> pd.DataFrame:
    """Admission-hour features for every patient (same recipe as training)."""
    patients = pd.read_csv(RAW / "patients.csv")
    vitals = pd.read_csv(RAW / "vitals_timeseries.csv")
    labs = pd.read_csv(RAW / "labs_timeseries.csv")
    h0 = vitals[vitals["hour_from_admission"] == 0]
    vcols = ["heart_rate", "respiratory_rate", "spo2_pct", "temperature_c",
             "systolic_bp", "diastolic_bp", "oxygen_flow", "mobility_score",
             "nurse_alert"]
    v0 = h0.set_index("patient_id")[vcols].rename(
        columns={c: f"{c}_adm" for c in vcols})
    lcols = ["wbc_count", "lactate", "creatinine", "crp_level", "hemoglobin"]
    l0 = labs[labs["hour_from_admission"] == 0].set_index("patient_id")[lcols].rename(
        columns={c: f"{c}_adm" for c in lcols})
    rank = {"none": 0, "nasal": 1, "mask": 2, "hfnc": 3, "niv": 4}
    device = h0.assign(dev=h0["oxygen_device"].map(rank).fillna(0)).groupby(
        "patient_id")["dev"].max().rename("oxygen_support_adm")
    df = patients.set_index("patient_id").join(v0).join(l0).join(device)
    return df


def risk_levels(prob: float, thresholds: dict) -> str:
    if prob < thresholds["low_max"]:
        return "LOW"
    if prob < thresholds["medium_max"]:
        return "MEDIUM"
    return "HIGH"


def is_critical(prob: float, thresholds: dict) -> bool:
    return prob >= thresholds.get("critical_min", 1.0)


def _label(prob: float, level: str, thresholds: dict) -> str:
    if level == "HIGH" and is_critical(prob, thresholds):
        return "Critical"
    return {"LOW": "Low", "MEDIUM": "Watch", "HIGH": "High"}[level]


def _predict_batch(frame: pd.DataFrame) -> np.ndarray:
    c = _load()
    cols = c["features"]["numerical"] + c["features"]["categorical"]
    raw = c["model"].predict_proba(frame[cols])[:, 1]
    return np.asarray(c["calibrator"].transform(raw), dtype=float)


def _predict_row(row: pd.Series) -> float:
    return float(_predict_batch(pd.DataFrame([row]))[0])


def contributing_factors(row: pd.Series, prob: float, top_n: int = 4) -> list[str]:
    """Rank the patient's most abnormal high-importance features and phrase
    each factor from the actual value vs cohort p10/p90 reference."""
    c = _load()
    ref = c["reference"]["cohort_stats"]
    imp = c["reference"]["feature_importance"]
    scored = []
    for feat, importance in imp.items():
        if feat not in ref or feat not in row or pd.isna(row[feat]):
            continue
        val = float(row[feat])
        stats = ref[feat]
        # Distance beyond the cohort's normal band, in units of the band width.
        band = max(stats["p90"] - stats["p10"], 1e-9)
        if val > stats["p90"]:
            dev = (val - stats["p90"]) / band
            direction = "high"
        elif val < stats["p10"]:
            dev = (stats["p10"] - val) / band
            direction = "low"
        else:
            continue  # inside normal cohort range -> not a useful explanation
        bad = HIGH_IS_BAD.get(feat, True)
        risk_increasing = (direction == "high" and bad) or (direction == "low" and not bad)
        if not risk_increasing:
            continue
        scored.append((importance * dev, feat, val, direction, stats))
    scored.sort(key=lambda t: -t[0])
    out = []
    for _, feat, val, direction, stats in scored[:top_n]:
        label = FEATURE_LABELS.get(feat, feat.replace("_adm", "").replace("_", " "))
        shown = f"{val:.1f}" if abs(val) >= 1 else f"{val:.2f}"
        out.append(
            f"{label} {direction} at admission ({shown}; cohort typical "
            f"{stats['p10']:.1f}-{stats['p90']:.1f})")
    if not out:
        out.append(
            "All admission measurements fall inside the typical cohort range; "
            "risk is driven by the combined pattern across features.")
    return out


def score_patient(patient_id: int) -> dict | None:
    """Full risk record for one patient (model output, no invented fields)."""
    c = _load()
    frame = c["frame"]
    if patient_id not in frame.index:
        return None
    row = frame.loc[patient_id]
    prob = _predict_row(row)
    level = risk_levels(prob, c["meta"]["thresholds"])
    label = _label(prob, level, c["meta"]["thresholds"])
    return {
        "patient_id": int(patient_id),
        "age": int(row["age"]),
        "gender": row["gender"],
        "admission_type": row["admission_type"],
        "comorbidity_index": int(row["comorbidity_index"]),
        "risk_probability": round(prob, 4),
        "risk_pct": round(prob * 100, 1),
        "risk_level": level,
        "risk_label": label,
        "prediction_window": c["meta"]["prediction_window"],
        "factors": contributing_factors(row, prob),
        "synthetic_data_note": (
            "Decision-support risk estimate from synthetic clinical data, "
            "for demonstration purposes only."),
    }


def score_many(patient_ids=None) -> dict[int, dict]:
    """Batch-score patients (one vectorized predict_proba call).

    patient_ids=None scores the full cohort. Returns {patient_id: record}.
    """
    c = _load()
    frame = c["frame"]
    if patient_ids is not None:
        frame = frame[frame.index.isin(set(patient_ids))]
    probs = _predict_batch(frame)
    thr = c["meta"]["thresholds"]
    out = {}
    for pid, prob in zip(frame.index, probs):
        row = frame.loc[pid]
        prob = float(prob)
        level = risk_levels(prob, thr)
        label = _label(prob, level, thr)
        out[int(pid)] = {
            "patient_id": int(pid),
            "age": int(row.age),
            "gender": row.gender,
            "admission_type": row.admission_type,
            "comorbidity_index": int(row.comorbidity_index),
            "risk_probability": round(prob, 4),
            "risk_pct": round(prob * 100, 1),
            "risk_level": level,
            "risk_label": label,
            "prediction_window": c["meta"]["prediction_window"],
            "factors": contributing_factors(row, prob),
            "synthetic_data_note": (
                "Decision-support risk estimate from synthetic clinical data, "
                "for demonstration purposes only."),
        }
    return out


def list_patients(top_n: int = 25, min_level: str | None = None) -> list[dict]:
    """Highest-risk patients first (batch-scored once, then cached)."""
    c = _load()
    if "scores" not in c:
        c["scores"] = pd.Series(_predict_batch(c["frame"]),
                                index=c["frame"].index)
    scores = c["scores"]
    thr = c["meta"]["thresholds"]
    order = scores.sort_values(ascending=False).index
    out = []
    for pid in order:
        level = risk_levels(float(scores[pid]), thr)
        if min_level:
            rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            if rank[level] < rank.get(min_level, 0):
                continue
        rec = score_patient(int(pid))
        out.append(rec)
        if len(out) >= top_n:
            break
    return out


def risk_summary() -> dict:
    """Cohort-level counts + model metrics for the dashboard/risk page."""
    c = _load()
    thr = c["meta"]["thresholds"]
    scores = c.get("scores")
    if scores is None:
        scores = pd.Series(_predict_batch(c["frame"]), index=c["frame"].index)
        c["scores"] = scores
    levels = scores.map(lambda p: risk_levels(p, thr))
    m = c["metrics"]
    return {
        "total_patients": int(len(scores)),
        "high_risk": int((levels == "HIGH").sum()),
        "medium_risk": int((levels == "MEDIUM").sum()),
        "low_risk": int((levels == "LOW").sum()),
        "critical_risk": int(scores.map(lambda p: is_critical(p, thr)).sum()),
        "mean_risk": round(float(scores.mean()) * 100, 2),
        "thresholds": thr,
        "model_metrics": {
            "algorithm": m.get("selected_model"),
            "roc_auc": m.get("roc_auc"),
            "pr_auc": m.get("pr_auc"),
            "recall": m.get("recall"),
            "precision": m.get("precision"),
            "band_performance": m.get("band_performance"),
        },
        "prediction_window": c["meta"]["prediction_window"],
        "data_note": c["meta"]["data_source"],
    }
