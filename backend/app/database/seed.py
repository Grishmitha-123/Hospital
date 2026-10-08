"""Seed the PostgreSQL database from the raw datasets.

Idempotent: clears and reloads rows it owns. Run with `python -m app.database.seed`.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sqlalchemy import delete

from ..services import forecast_service, risk_service
from ..config import DATABASE_URL
from .models import (DemandHistory, PatientLab, PatientRecord, PatientVital,
                     Resource)
from .session import Base, SessionLocal, engine

RAW_OPS = (Path(__file__).resolve().parents[2] / "data" / "raw" /
           "Bed-prediction-main" / "data" / "processed")
RAW_CLIN = (Path(__file__).resolve().parents[2] / "data" / "raw" /
            "hospital-deterioration-dataset-main" / "data")

# "Now" operational capacity for the demonstration hospital.
# Occupancy is derived from the latest observed day of the ops dataset;
# totals are this hospital's configured capacity. ED capacity is throughput
# in arrivals/day (the dataset reports absolute arrivals/day).
CAPACITY = {
    "general_beds": {"category": "bed", "total": 100},
    "icu_beds": {"category": "bed", "total": 30},
    "emergency_capacity": {"category": "ed", "total": 120},
    "staff": {"category": "staffing", "total": 80},
    "or_theatres": {"category": "theatre", "total": 10},
}


def seed_resources(db: SessionLocal) -> None:
    ops = forecast_service.current_state_row()
    occupancy = float(ops["Bed_Occupancy"]) / 100.0
    now = forecast_service.forecast(24)
    # occupied = beds in use (beds), arrivals handled today (ED),
    # staff on duty (staffing), theatres booked (theatre).
    values = {
        "general_beds": int(round(CAPACITY["general_beds"]["total"] * occupancy)),
        "icu_beds": int(round(CAPACITY["icu_beds"]["total"] * occupancy)),
        "emergency_capacity": int(now["emergency"][0]),
        "staff": int(round(CAPACITY["staff"]["total"] * 0.85)),
        "or_theatres": int(round(CAPACITY["or_theatres"]["total"] * 0.6)),
    }
    db.execute(delete(Resource))
    for name, cfg in CAPACITY.items():
        db.add(Resource(name=name, category=cfg["category"],
                        total=cfg["total"], occupied=values[name]))
    db.commit()
    print("resources seeded:", values)


def seed_demand_history(db: SessionLocal) -> None:
    df = forecast_service.load_models()["history"]
    db.execute(delete(DemandHistory))
    for _, r in df.iterrows():
        db.add(DemandHistory(
            date=r["Date"].strftime("%Y-%m-%d"),
            icu_demand=int(r["ICU_Demand"]),
            emergency_demand=int(r["Emergency_Demand"]),
            general_demand=int(r["General_Demand"]),
            bed_occupancy=float(r["Bed_Occupancy"]),
            staff_count=float(r["Staff_Count"]),
            flu_cases=int(r["Flu_Cases"]),
        ))
    db.commit()
    print("demand_history rows:", len(df))


def seed_patients(db: SessionLocal, sample: int = 0) -> None:
    """Load patients + first-12h vitals/labs, with cached model scores.

    sample=0 loads the full 10k cohort; sample>0 loads a subset (fast demo).
    """
    t0 = time.time()
    patients = pd.read_csv(RAW_CLIN / "patients.csv")
    vitals = pd.read_csv(RAW_CLIN / "vitals_timeseries.csv")
    labs = pd.read_csv(RAW_CLIN / "labs_timeseries.csv")
    if sample:
        ids = set(patients["patient_id"].head(sample))
        patients = patients[patients["patient_id"].isin(ids)]
        vitals = vitals[vitals["patient_id"].isin(ids)]
        labs = labs[labs["patient_id"].isin(ids)]

    # Cached risk scores come straight from the trained model service
    # (one vectorized batch prediction for the whole cohort).
    summary = risk_service.risk_summary()
    scored = risk_service.score_many(patients["patient_id"].tolist())

    db.execute(delete(PatientVital))
    db.execute(delete(PatientLab))
    db.execute(delete(PatientRecord))
    for _, r in patients.iterrows():
        pid = int(r["patient_id"])
        cached = scored.get(pid) or {}
        db.add(PatientRecord(
            patient_id=pid, age=int(r["age"]), gender=r["gender"],
            admission_type=r["admission_type"],
            comorbidity_index=int(r["comorbidity_index"]),
            risk_probability=cached.get("risk_probability"),
            risk_level=cached.get("risk_level"),
            factors=cached.get("factors"),
        ))
    db.flush()

    v = vitals[vitals["hour_from_admission"] < 12]
    db.execute(PatientVital.__table__.insert(), [
        {"patient_id": int(r.patient_id),
         "hour_from_admission": int(r.hour_from_admission),
         "heart_rate": float(r.heart_rate),
         "respiratory_rate": float(r.respiratory_rate),
         "spo2_pct": float(r.spo2_pct),
         "temperature_c": float(r.temperature_c),
         "systolic_bp": float(r.systolic_bp),
         "diastolic_bp": float(r.diastolic_bp),
         "oxygen_device": r.oxygen_device,
         "oxygen_flow": float(r.oxygen_flow),
         "mobility_score": int(r.mobility_score),
         "nurse_alert": int(r.nurse_alert)}
        for r in v.itertuples(index=False)])
    l = labs[labs["hour_from_admission"] < 12]
    db.execute(PatientLab.__table__.insert(), [
        {"patient_id": int(r.patient_id),
         "hour_from_admission": int(r.hour_from_admission),
         "wbc_count": float(r.wbc_count), "lactate": float(r.lactate),
         "creatinine": float(r.creatinine), "crp_level": float(r.crp_level),
         "hemoglobin": float(r.hemoglobin)}
        for r in l.itertuples(index=False)])
    db.commit()
    print(f"patients={len(patients)} vitals={len(v)} labs={len(l)} "
          f"in {time.time()-t0:.1f}s | cohort high-risk={summary['high_risk']}")


def main():
    print("DATABASE_URL:", DATABASE_URL)
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        seed_resources(db)
        seed_demand_history(db)
        seed_patients(db)
    finally:
        db.close()
    print("seed complete")


if __name__ == "__main__":
    main()
