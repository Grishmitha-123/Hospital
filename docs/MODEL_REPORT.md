# MediCare AI — Model Report

All models were **trained by this project** from the two supplied datasets via
`backend/training/`. Pre-existing artifacts in the source repos
(`icu_xgboost.pkl`, `emergency_prophet.json`, `general_randomforest.pkl`) were
**not** used or presented as ours.

> **Data notice:** both datasets are synthetic/simulated. The hospital
> deterioration dataset is explicitly a simulated cohort; the bed-prediction
> data is simulated hospital-operations data. Everything below is for
> development and demonstration — not real patients, not clinical advice.

---

## 1. Patient Deterioration Risk Model

- **Script:** `python training/train_patient_risk.py`
- **Artifacts:** `models/patient_risk_model.pkl` (model + isotonic calibrator),
  `feature_columns.json`, `metrics.json`, `model_metadata.json`,
  `risk_reference.json`
- **Target:** `deterioration_within_12h_from_admission` (0/1). Prevalence 3.09%.
- **Prediction window:** deterioration within 12h **from admission**, using only
  admission-hour (hour 0) information.
- **Features:** static (age, gender, comorbidity index, admission type) plus
  hour-0 vitals (HR, RR, SpO2, temp, SBP, DBP, oxygen flow, mobility, nurse
  alert), hour-0 labs (WBC, lactate, creatinine, CRP, hemoglobin) and peak
  oxygen-support level at admission.
- **Excluded to avoid leakage:** `deterioration_event`, `deterioration_hour`,
  `deterioration_within_12h_from_admission`, `los_hours`, plus the latent
  simulation parameters `baseline_risk_score` and `sepsis_risk_score` (a real
  hospital would not measure these; they encode the generated outcome).
- **Candidates compared:** Logistic Regression, Random Forest,
  HistGradientBoosting. Selection by validation PR-AUC (imbalanced target).
- **Selected:** Random Forest (400 trees, `class_weight=balanced_subsample`),
  with isotonic probability calibration on a held-out calibration split.
- **Split:** patient-level stratified 70/15/15 fit/calibration/test (6800/1200/2000).

### Test metrics (threshold 0.5, calibration applied)

| metric | value |
|---|---|
| ROC-AUC | 0.764 |
| PR-AUC | 0.112 (baseline 0.031) |
| accuracy | 0.969 |

At the default 0.5 cut the (uncalibrated-threshold) precision/recall collapse to
0 because calibrated probabilities at 3% prevalence rarely exceed 0.5 — which
is exactly why risk **bands** are used instead of a single cut:

| band | share of patients | precision | recall |
|---|---|---|---|
| HIGH (top ~15%) | 14.3% | 0.102 | 0.468 |
| HIGH or MEDIUM (top ~37%) | 36.6% | 0.064 | 0.758 |

Interpretation: the HIGH band concentrates ~3.3× the base rate of true
deteriorations; HIGH+MEDIUM captures ~76% of all deteriorations while flagging
~37% of patients. Missing a deteriorating patient is costlier than an extra
warning, so thresholds are set to keep recall high rather than precision.

### Thresholds (documented, calibration-derived)

- `low_max` = 0.0351 (60th percentile of calibrated risk, capped 0.30)
- `medium_max` = 0.0759 (85th percentile, capped 0.70)
- `critical_min` = 0.0833 (smallest distinct calibrated value above the HIGH cut
  flagging ≤5% of the calibration cohort; isotonic calibration produces tied
  probabilities so a plain percentile can land inside a tie block)
- `LOW < low_max ≤ MEDIUM < medium_max ≤ HIGH`; `prob ≥ critical_min` also
  flags CRITICAL.

### Contributing factors

Per-patient explanations rank the model's most important features by how far the
patient's admission values fall outside the cohort's typical p10–p90 band,
phrased with the actual value and cohort range (e.g. "lactate high at admission
(4.1; cohort typical 0.8–2.3)"). Global importance comes from the trained
forest; band statistics come from the training cohort — no invented reasons.

### Limitations

- Synthetic data: real-world discrimination will be lower.
- Single admission-time snapshot; no trend information after hour 0 by design
  (avoids overlapping the 12h target window).
- 3% prevalence makes absolute precision low even in the HIGH band.
- Decision-support estimates for demonstration only — not a clinical device.

---

## 2. Demand Forecasting Models (ICU / Emergency / General beds)

- **Script:** `python training/train_demand_models.py`
- **Artifacts:** `models/icu_demand_model.pkl`,
  `models/emergency_demand_model.pkl`, `models/general_beds_demand_model.pkl`
  (each holds one model per horizon), `forecast_metrics.json`,
  `forecast_metadata.json`
- **Data:** `final_hospital_forecasting_dataset.csv` — 208 consecutive daily
  hospital-operations records (synthetic).
- **Approach:** direct multi-horizon regression — one Random Forest per
  (series × horizon), horizons 1/2/3 days = 24/48/72h. A stacked-horizon pass
  selects the algorithm; per-horizon models are then refit on train+validation.
- **Features:** target lags 1/2/3/7/14, rolling mean/std (3,7), day-of-week,
  month, and day-t exogenous signals (staff count, flu cases, occupancy,
  discharges, admissions) — all observed at forecast time; only targets are
  lagged so no future values ever leak.
- **Split:** chronological 70/15/15 by forecast-origin date (no shuffling).

### Test metrics (latest run)

| series | algorithm | 24h MAE / R² | 48h MAE / R² | 72h MAE / R² |
|---|---|---|---|---|
| ICU | Random Forest | 3.48 / 0.35 | 4.11 / 0.16 | 4.48 / −0.04 |
| Emergency | Random Forest | 4.65 / 0.36 | 5.60 / 0.15 | 5.95 / −0.01 |
| General beds | Random Forest | 15.15 / 0.34 | 17.75 / 0.17 | 19.19 / −0.01 |

### Baseline comparison (naive persistence on the test window)

| series | naive MAE | model MAE (aggregate) |
|---|---|---|
| ICU | 4.94 | 4.03 |
| Emergency | 6.63 | 5.40 |
| General beds | 21.37 | 17.36 |

The models beat persistence on every series, but R² degrades with horizon and
turns slightly negative at 72h: this synthetic series has limited predictable
structure beyond a few days and the test window shifts regime (flu peak in the
validation window). We report this honestly rather than hiding it.

### Limitations

- 208 daily rows is a small training set; uncertainty grows with horizon.
- No external covariates (weather, holidays, outbreaks) beyond flu cases.
- Projections are in dataset demand units; the API maps them onto the demo
  hospital's configured capacity by applying the model's growth ratio to the
  observed census (`state_service.get_demand`).

---

## 3. Resource Optimizer (Google OR-Tools — not an ML model)

- **Code:** `app/optimization/solver.py`, `app/optimization/explanation.py`
- **Solver:** OR-Tools CP-SAT (integer program, 15s limit).

Objective (minimize):

```
100·ICU shortfall + 60·Emergency shortfall + 40·General-bed shortfall
+ 25·3·Staff shortfall + 25·Staff overload
+ 80·Reserve below minimum + 1·Reserve/ OR unused above minimum
```

Hard constraints: allocation + reserve ≤ available ICU; allocation ≤ ED
capacity / free general beds / staff on duty / OR capacity; staff ≥ safe minimum
when achievable; all variables integer and non-negative.

**Outputs are computed by the solver** — no allocation, shortage or reserve
number in the UI is hardcoded. Explanations quote the actual forecast demand,
current capacity and solver values.
