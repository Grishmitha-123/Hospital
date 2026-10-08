# 🏥 Hospital Deterioration Dataset — Simulated Early Warning Benchmark

High-fidelity **simulated hospital cohort**, ML-ready for building and benchmarking **early warning systems** and **clinical deterioration risk models** using hourly time series.

---

## 📌 1. TL;DR

- **10,000 simulated hospital admissions**, each followed for up to **72 hours**  
- **Hourly vitals + labs** joined with rich patient-level context  
- Multiple deterioration outcomes, including a clean **“next 12 hours”** window label  
- Designed as an **ML-ready benchmark** for early warning, risk modeling, and time-series methods  
- All records are **fully simulated** and **do not correspond to real patients**

This repo mirrors the dataset published on Kaggle / Hugging Face and provides a clean, version-controlled home for the CSV files and documentation.

---

## 🧠 2. Overview

This repository contains a high-resolution simulated hospital cohort for **early clinical deterioration modeling**.

Each admission is followed for up to **72 hours** from arrival, with:

- Hourly **vital signs**  
  Heart rate, blood pressure, respiratory rate, SpO₂, temperature, oxygen support, mobility, nurse alerts.
- Hourly **laboratory values**  
  WBC, lactate, creatinine, CRP, hemoglobin, sepsis risk.
- Patient-level **context**  
  Demographics, comorbidity burden, admission type, baseline risk, and length of stay.
- Multiple **deterioration outcomes**, including a precisely defined **“next 12 hours”** prediction target.

All records are **fully simulated** and **internally consistent**  
(e.g., oxygen flow is always `0.0` when no oxygen device is used).

The dataset is designed as a **machine-learning-ready benchmark** for:

- Early warning systems (EWS) and rapid response triggers  
- Sepsis / deterioration risk modeling and score calibration  
- Time-series and sequence models (RNNs, Temporal CNNs, Transformers, temporal tabular models)  
- Teaching, prototyping, and methods research in clinical ML, without touching real-patient data

For column-level details, see the **data dictionary**:

- `data_dictionary.md`

## ✅ Integrity & validation

This repo includes two lightweight safety nets:

- **Checksums** (`checksums.sha256`) so you can verify files were not modified.
- **Validation script** (`scripts/validate_dataset.py`) for key/range/alignment checks.

Run locally:

```bash
# Verify published files (requires sha256sum)
sha256sum -c checksums.sha256

# Cross-platform verification
python scripts/make_checksums.py --check

# Dataset sanity checks
python scripts/validate_dataset.py --data-dir data

# Build derived views
python scripts/build_views.py --data-dir data --out-dir generated
```

Tip: add `--strict` to treat soft data issues as errors (default is warning-only for a small set of known edge cases).


Windows PowerShell checksum example:

```powershell
Get-FileHash .\data\patients.csv -Algorithm SHA256
```

---

## 📂 3. Dataset contents

### 3.1 📄 Files

| File | Granularity | Description |
|---|---|---|
| `data/patients.csv` | 1 row per patient (10,000 rows) | Static patient data: demographics, comorbidities, admission type, baseline risk, LOS, and outcomes |
| `data/vitals_timeseries.csv` | 1 row per `(patient_id, hour_from_admission)` | Hourly vital signs and monitoring fields |
| `data/labs_timeseries.csv` | 1 row per `(patient_id, hour_from_admission)` | Hourly lab values and sepsis risk score |
| `data_dictionary.md` | Documentation | Column-level dictionary (includes canonical tables + derived views) |
| `generated/hospital_deterioration_hourly_panel.csv` | 1 row per `(patient_id, hour_from_admission)` | **Generated**: joined vitals + labs + patient features + labels |
| `generated/hospital_deterioration_ml_ready.csv` | 1 row per hourly observation | **Generated**: features + target `deterioration_next_12h` |

> Note: The `generated/` views are produced by `python scripts/build_views.py` and are **not committed** to keep the repo lightweight.

All features are fully observed (**no missing values**). Time is expressed as **hours from admission**.

---

### 3.2 ⏱️ Granularity

- **Patient level**  
  - `data/patients.csv` — one row per patient (10,000 rows).

- **Hourly time series**  
  - `data/vitals_timeseries.csv` — one row per `(patient_id, hour_from_admission)` for vital signs.  
  - `data/labs_timeseries.csv` — one row per `(patient_id, hour_from_admission)` for lab values.  
  - `generated/hospital_deterioration_hourly_panel.csv` — one row per `(patient_id, hour_from_admission)` with vitals, labs, static features, and labels.  
  - `generated/hospital_deterioration_ml_ready.csv` — same hourly granularity, but with features + single target only.

Each patient has a length of stay between **12 and 72 hours** (`los_hours`), and the time series cover:

```text
hour_from_admission = 0, 1, 2, ..., los_hours - 1
```

---

## 📘 4. File descriptions

### 4.1 `data/patients.csv` — Patient-level static data

**One row per patient (10,000 rows).**  
Includes:

- `patient_id` — artificial identifier  
- `age`, `gender`  
- `comorbidity_index` — aggregate comorbidity burden  
- `admission_type` — ED / Elective / Transfer  
- `baseline_risk_score` — latent baseline deterioration risk (0–1)  
- `los_hours` — length of stay (12–72 hours)  
- Deterioration outcomes:
  - `deterioration_event`
  - `deterioration_within_12h_from_admission`
  - `deterioration_hour` (or -1 if no event)

✅ **Use this for:**

- Cohort summaries and descriptive statistics  
- Patient-level stratification and outcome prevalence  
- Linking to hourly tables via `patient_id`.

---

### 4.2 `data/vitals_timeseries.csv` — Hourly vital signs

**One row per `(patient_id, hour_from_admission)`.**

Includes hourly:

- `heart_rate`, `respiratory_rate`  
- `spo2_pct`, `temperature_c`  
- `systolic_bp`, `diastolic_bp`  
- `oxygen_device`, `oxygen_flow`  
- `mobility_score`  
- `nurse_alert`

**Consistency note:** when `oxygen_device == "none"`, `oxygen_flow` is always `0.0`.

✅ **Use this for:**

- Pure vitals time-series modeling  
- Feature engineering on dynamic physiologic trends.

---

### 4.3 `data/labs_timeseries.csv` — Hourly lab values

**One row per `(patient_id, hour_from_admission)`.**

Includes hourly:

- `wbc_count`  
- `lactate`  
- `creatinine`  
- `crp_level`  
- `hemoglobin`  
- `sepsis_risk_score` — latent sepsis risk (0–1)

✅ **Use this for:**

- Lab-centric modeling  
- Combining vitals + labs for richer time-series models.

---

### 4.4 `generated/hospital_deterioration_hourly_panel.csv` — Full joined hourly panel

**One row per `(patient_id, hour_from_admission)`** with:

- Hourly **vitals**  
- Hourly **labs**  
- Static patient-level features:
  - `age`, `gender`, `comorbidity_index`, `admission_type`, `los_hours`, `baseline_risk_score`
- All deterioration labels:
  - `deterioration_event`
  - `deterioration_hour`
  - `deterioration_within_12h_from_admission`
  - `deterioration_next_12h` (window label)

✅ **Use this when you want a single “wide” table for:**

- Custom label definitions  
- Multi-task learning  
- Advanced feature engineering without doing your own joins.

---

### 4.5 `generated/hospital_deterioration_ml_ready.csv` — ML-ready classification table

**One row per hourly observation** (per patient and `hour_from_admission`).

This file is a clean modeling view:

- **Features only**:
  - Hourly vitals and labs
  - Static covariates at admission (`age`, `gender`, `comorbidity_index`, `admission_type`)
- **Single target**:
  - `deterioration_next_12h` (0/1)

This is the **recommended entry point** for most users.

Minimal example:

```python
import pandas as pd

df = pd.read_csv("generated/hospital_deterioration_ml_ready.csv")

X = df.drop(columns=["deterioration_next_12h"])
y = df["deterioration_next_12h"]

print(X.shape, y.mean())
```

---

## 🎯 5. Main prediction task: deterioration in the next 12 hours

For each row (one patient at a specific `hour_from_admission = t`), the binary target:

```text
deterioration_next_12h
```

is defined using the patient-level `deterioration_hour` as:

- `deterioration_next_12h = 1`  
  if a deterioration event occurs **after the current hour** and **within the next 12 hours**:

  ```text
  t < deterioration_hour ≤ t + 12
  ```

- `deterioration_next_12h = 0` otherwise, including:
  - stays with no deterioration at all, and  
  - the hour where the event is happening now.

This framing mirrors how **operational early-warning systems** are deployed: models predict risk ahead of time, providing a buffer for clinicians to intervene.

It is well-suited for:

- Early-warning models with different alert thresholds  
- Cost-sensitive optimization (balancing missed events vs false alarms)  
- Temporal evaluation (patient-level splits, rolling validation, horizon-based metrics)

---

## 🧪 6. Example usage

### 6.1 Load ML-ready dataset

```python
import pandas as pd

ml = pd.read_csv("generated/hospital_deterioration_ml_ready.csv")

X = ml.drop(columns=["deterioration_next_12h"])
y = ml["deterioration_next_12h"]

print(f"Features: {X.shape}, Target positive rate: {y.mean():.3f}")
```

### 6.2 Join vitals, labs, and patients

```python
import pandas as pd

patients = pd.read_csv("data/patients.csv")
vitals = pd.read_csv("data/vitals_timeseries.csv")
labs = pd.read_csv("data/labs_timeseries.csv")

panel = (
    vitals
    .merge(labs, on=["patient_id", "hour_from_admission"], how="inner")
    .merge(patients, on="patient_id", how="left")
)

print(panel.shape)
```

You can also start from `generated/hospital_deterioration_hourly_panel.csv` directly if you prefer a pre-joined view.

---

## 🧭 7. Suggested ML tasks & use cases

Some possible tasks:

- **Binary classification**  
  - Predict `deterioration_next_12h` using `generated/hospital_deterioration_ml_ready.csv`.

- **Time-series and sequence modeling**  
  - Use recurrent or transformer models over the hourly series for each patient.

- **Risk score calibration and interpretability**  
  - Compare classical logistic regression, tree-based models, and deep models.  
  - Study calibration curves and feature importance (e.g., SHAP).

- **Threshold and policy optimization**  
  - Explore different alert thresholds and their impact on recall, precision, and false alarm burden.

- **Teaching and benchmarking**  
  - Demonstrate end-to-end pipelines: data loading → feature engineering → modeling → evaluation.

---

## 🔐 8. Data generation & privacy

The cohort is generated via a **high-fidelity, privacy-preserving simulation pipeline**:

- Patient profiles (age, comorbidities, admission route) determine a **baseline risk** of deterioration.  
- Vital and lab trajectories evolve over time, with **higher-risk patients drifting towards more abnormal values** as they approach deterioration.  
- Latent risk scores:
  - `baseline_risk_score` (at admission)
  - `sepsis_risk_score` (hourly)

  are derived from the same underlying factors and can be used for:

  - Risk calibration  
  - Feature importance analysis  
  - Interpretable modeling.

No row corresponds to a real individual or a specific hospital.  
The dataset is intended for **methods research**, **benchmarking**, **education**, and **prototyping** early warning systems, not for real-time clinical decision-making.

---

## ⚠️ 9. Limitations & ethical notes

- The data are **fully synthetic** and cannot be used to make clinical decisions about real patients.  
- Patterns are designed to be **plausible**, not to reproduce any specific hospital or population.  
- There are no missing values: this is a design choice to keep the benchmark clean and focused on modeling, not imputation.  
- Researchers should treat this as a **simulation benchmark**, not a substitute for real-world validation.

---

## 📜 10. License

Unless otherwise stated, the dataset is released under:

**CC BY 4.0 — Creative Commons Attribution 4.0 International**

You are free to:

- Share — copy and redistribute the material in any medium or format  
- Adapt — remix, transform, and build upon the material for any purpose, even commercially  

Under the following terms:

- Attribution — give appropriate credit to the original author and provide a link to the license.

More details: <https://creativecommons.org/licenses/by/4.0/>

---

## ✍️ 11. Attribution

**Author:** Tarek Masryo  

If you use this dataset in a paper, blog post, notebook, or teaching material,  
please consider citing the repository and mentioning:

> “Hospital Deterioration Dataset — Simulated Early Warning Benchmark (Tarek Masryo)”
