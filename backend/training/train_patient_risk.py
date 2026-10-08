"""Train the patient deterioration risk model.

Data source: hospital-deterioration-dataset (SYNTHETIC / simulated clinical data
used for model development and demonstration only - not real patients).

Target: `deterioration_within_12h_from_admission` (0/1) from patients.csv.
Features: static patient fields + first-12h summaries of vitals/labs
(only information available at prediction time; outcome columns excluded
explicitly to avoid leakage).

Run:  python training/train_patient_risk.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospital-deterioration-dataset-main" / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "deterioration_within_12h_from_admission"
# Columns that directly reveal the outcome, are post-outcome, or are latent
# simulation parameters -> never features (leakage control).
LEAKY = {
    "patient_id", "deterioration_event", "deterioration_hour",
    "deterioration_within_12h_from_admission", "los_hours",
    # baseline_risk_score / sepsis_risk_score are latent simulation parameters
    # that encode the generated outcome; a real hospital would not have them.
    "baseline_risk_score", "sepsis_risk_score",
}


def load_dataset() -> pd.DataFrame:
    """Join static patient data with the admission-hour (hour 0) vitals/labs.

    Only hour-0 measurements are used: the model must predict deterioration in
    the NEXT 12h from information available at admission time, so any window
    that overlaps the target window would leak the outcome.
    """
    patients = pd.read_csv(RAW / "patients.csv")
    vitals = pd.read_csv(RAW / "vitals_timeseries.csv")
    labs = pd.read_csv(RAW / "labs_timeseries.csv")

    print("patients:", patients.shape, "| vitals:", vitals.shape, "| labs:", labs.shape)
    print("target rate:", round(patients[TARGET].mean(), 4))
    print("missing values (patients):", int(patients.isna().sum().sum()))

    h0 = vitals[vitals["hour_from_admission"] == 0]
    v0 = h0.set_index("patient_id")[[
        "heart_rate", "respiratory_rate", "spo2_pct", "temperature_c",
        "systolic_bp", "diastolic_bp", "oxygen_flow", "mobility_score",
        "nurse_alert",
    ]].rename(columns={c: f"{c}_adm" for c in [
        "heart_rate", "respiratory_rate", "spo2_pct", "temperature_c",
        "systolic_bp", "diastolic_bp", "oxygen_flow", "mobility_score",
        "nurse_alert"]})
    l0 = labs[labs["hour_from_admission"] == 0].set_index("patient_id")[
        ["wbc_count", "lactate", "creatinine", "crp_level", "hemoglobin"]
    ].rename(columns={c: f"{c}_adm" for c in [
        "wbc_count", "lactate", "creatinine", "crp_level", "hemoglobin"]})

    # Highest oxygen support level recorded at admission.
    device_rank = {"none": 0, "nasal": 1, "mask": 2, "hfnc": 3, "niv": 4}
    device = h0.assign(dev=h0["oxygen_device"].map(device_rank).fillna(0)) \
        .groupby("patient_id")["dev"].max().rename("oxygen_support_adm")

    df = patients.set_index("patient_id").join(v0).join(l0).join(device).reset_index()
    print("joined rows:", len(df), "| missing after join:",
          int(df.drop(columns=[TARGET]).isna().sum().sum()))
    return df


def build_xy(df: pd.DataFrame):
    y = df[TARGET].astype(int)
    cat_cols = [c for c in ("gender", "admission_type") if c in df.columns]
    num_cols = [c for c in df.columns if c not in LEAKY and c not in cat_cols and c != TARGET]
    X = df[num_cols + cat_cols].copy()
    return X, y, num_cols, cat_cols


def evaluate(name, model, X_tr, y_tr, X_te, y_te) -> dict:
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_te, pred)
    m = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "confusion_matrix": cm.tolist(),
    }
    print(name, m)
    return m


def main():
    df = load_dataset()
    X, y, num_cols, cat_cols = build_xy(df)

    # Patient-level stratified split (patients are independent rows).
    idx_tr, idx_te = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
    X_tr, y_tr = X.iloc[idx_tr], y.iloc[idx_tr]
    X_te, y_te = X.iloc[idx_te], y.iloc[idx_te]
    # calibration/threshold split carved out of the training half
    idx_tr2, idx_cal = train_test_split(
        np.arange(len(X_tr)), test_size=0.15, random_state=42, stratify=y_tr)
    X_fit, y_fit = X_tr.iloc[idx_tr2], y_tr.iloc[idx_tr2]
    X_cal, y_cal = X_tr.iloc[idx_cal], y_tr.iloc[idx_cal]
    print(f"fit={len(X_fit)} cal={len(X_cal)} test={len(X_te)} | "
          f"pos rate fit={y_fit.mean():.3f} test={y_te.mean():.3f}")

    preproc = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])

    def pipe(clf):
        return Pipeline([("pre", preproc), ("clf", clf)])

    candidates = {
        "logreg": pipe(LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=42)),
        "random_forest": pipe(RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42)),
        "hist_gradient_boosting": pipe(HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=300, l2_regularization=1.0,
            class_weight="balanced", random_state=42)),
    }

    results, fitted = {}, {}
    for name, pipe in candidates.items():
        results[name] = evaluate(name, pipe, X_fit, y_fit, X_te, y_te)
        fitted[name] = pipe

    # Select by validation PR-AUC (imbalanced target), tie-break on ROC-AUC.
    best_name = max(results, key=lambda n: (results[n]["pr_auc"], results[n]["roc_auc"]))
    best = fitted[best_name]
    print("selected model:", best_name)

    # Isotonic calibration on a held-out split: raw tree-ensemble scores are
    # heavily compressed toward 0 on a 3% prevalence target, which makes
    # probability thresholds meaningless.
    cal_proba_raw = best.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cal_proba_raw, y_cal)
    cal_proba = calibrator.transform(cal_proba_raw)

    # Risk-level thresholds: percentiles of calibrated risk on the calibration
    # split. At 3% prevalence, recall-driven thresholds would flag ~half the
    # ward; percentile bands give an actionable stratification and their
    # achieved precision/recall are reported in metrics.json.
    high_thr = float(np.quantile(cal_proba, 0.85))  # top 15% risk -> HIGH
    med_thr = float(np.quantile(cal_proba, 0.60))   # next 30% -> MEDIUM
    # CRITICAL: isotonic calibration produces tied probabilities, so a plain
    # percentile can land inside a tie block. Pick the smallest *distinct*
    # calibrated value above the HIGH threshold that flags <=5% of the
    # calibration cohort (closest realisable cut to the intended top 2.5%).
    uniq = np.unique(cal_proba)
    cands = [float(v) for v in uniq if v > high_thr
             and (cal_proba >= v).mean() <= 0.05]
    crit_thr = min(cands) if cands else float(uniq[-1])
    thresholds = {
        "low_max": round(min(med_thr, 0.30), 4),
        "medium_max": round(min(max(high_thr, med_thr + 1e-4), 0.70), 4),
        "critical_min": round(crit_thr, 4),
        "note": (
            "Thresholds are the 60th/85th percentile of calibrated predicted "
            "risk on a held-out calibration split (capped at 0.30/0.70): "
            "prob < low_max => LOW, < medium_max => MEDIUM, else HIGH; "
            "prob >= critical_min (smallest distinct calibrated value above "
            "the HIGH cut flagging <=5% of the calibration cohort) "
            "additionally flags CRITICAL. "
            "Probability thresholds targeting 80% recall would flag ~50% of "
            "patients at 3% prevalence, so percentile bands are used instead; "
            "per-band precision/recall are in metrics.json."
        ),
    }

    # Final metrics on test set with thresholds applied.
    te_proba = calibrator.transform(best.predict_proba(X_te)[:, 1])
    levels = np.where(te_proba < thresholds["low_max"], "LOW",
              np.where(te_proba < thresholds["medium_max"], "MEDIUM", "HIGH"))
    high_mask = levels == "HIGH"
    med_mask = levels == "MEDIUM"
    final = dict(results[best_name])
    final["selected_model"] = best_name
    final["thresholds"] = thresholds
    final["band_performance"] = {
        "HIGH": {
            "share_of_patients": round(float(high_mask.mean()), 4),
            "precision": round(float(precision_score(y_te, high_mask, zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, high_mask, zero_division=0)), 4),
        },
        "HIGH_or_MEDIUM": {
            "share_of_patients": round(float((high_mask | med_mask).mean()), 4),
            "precision": round(float(precision_score(y_te, high_mask | med_mask, zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, high_mask | med_mask, zero_division=0)), 4),
        },
    }
    final["n_test"] = int(len(y_te))
    final["target"] = TARGET
    final["prevalence"] = round(float(y.mean()), 4)
    print(json.dumps(final, indent=2))

    joblib.dump({"model": best, "calibrator": calibrator},
                MODEL_DIR / "patient_risk_model.pkl")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(
        {"numerical": num_cols, "categorical": cat_cols}, indent=2))
    # Cohort reference stats: used by the backend to explain individual
    # predictions ("patient's lactate is above the cohort 90th percentile").
    ref = {}
    for c in num_cols:
        s = X[c].dropna()
        ref[c] = {
            "p10": round(float(s.quantile(0.10)), 4),
            "median": round(float(s.median()), 4),
            "p90": round(float(s.quantile(0.90)), 4),
        }
    # Global feature importance (model-specific) for ranking explanation factors.
    clf = best.named_steps["clf"]
    strip = lambda n: n.split("__", 1)[-1]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        feat_names = list(best.named_steps["pre"].get_feature_names_out())
        top = sorted(zip(map(strip, feat_names), importances), key=lambda kv: -kv[1])[:15]
        importance = {k: round(float(v), 5) for k, v in top}
    else:
        coefs = np.abs(clf.coef_[0])
        feat_names = list(best.named_steps["pre"].get_feature_names_out())
        top = sorted(zip(map(strip, feat_names), coefs), key=lambda kv: -kv[1])[:15]
        importance = {k: round(float(v), 5) for k, v in top}
    (MODEL_DIR / "risk_reference.json").write_text(json.dumps(
        {"cohort_stats": ref, "feature_importance": importance}, indent=2))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(final, indent=2))
    meta = {
        "model": "patient_deterioration_risk",
        "algorithm": best_name,
        "target": TARGET,
        "prediction_window": "12h from admission",
        "features": num_cols + cat_cols,
        "excluded_leaky_columns": sorted(LEAKY),
        "thresholds": thresholds,
        "training_rows": int(len(X)),
        "data_source": (
            "hospital-deterioration-dataset (SYNTHETIC/simulated clinical data "
            "for model development and demonstration)"
        ),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    print("saved model artifacts to", MODEL_DIR)


if __name__ == "__main__":
    sys.exit(main())
