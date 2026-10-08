"""SQLAlchemy ORM models (PostgreSQL).

Tables kept deliberately small:
  patients / patient_vitals / patient_labs - clinical side (risk model input)
  resources                                - operational capacity ("now")
  demand_history                           - observed demand time series
  forecasts                                - saved model forecasts
  scenarios                                - what-if runs
  optimization_results                     - solver outputs + explanations
"""
from datetime import datetime

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, String,
                        Text)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .session import Base


class PatientRecord(Base):
    """Admission record + cached risk-model output for one patient."""
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String(8))
    admission_type: Mapped[str] = mapped_column(String(20))
    comorbidity_index: Mapped[int] = mapped_column(Integer)
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(12), nullable=True)
    factors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    vitals = relationship("PatientVital", back_populates="patient",
                          cascade="all, delete-orphan")
    labs = relationship("PatientLab", back_populates="patient",
                        cascade="all, delete-orphan")


class PatientVital(Base):
    __tablename__ = "patient_vitals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("patients.patient_id"), index=True)
    hour_from_admission: Mapped[int] = mapped_column(Integer)
    heart_rate: Mapped[float] = mapped_column(Float)
    respiratory_rate: Mapped[float] = mapped_column(Float)
    spo2_pct: Mapped[float] = mapped_column(Float)
    temperature_c: Mapped[float] = mapped_column(Float)
    systolic_bp: Mapped[float] = mapped_column(Float)
    diastolic_bp: Mapped[float] = mapped_column(Float)
    oxygen_device: Mapped[str] = mapped_column(String(12))
    oxygen_flow: Mapped[float] = mapped_column(Float)
    mobility_score: Mapped[int] = mapped_column(Integer)
    nurse_alert: Mapped[int] = mapped_column(Integer)

    patient = relationship("PatientRecord", back_populates="vitals")


class PatientLab(Base):
    __tablename__ = "patient_labs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("patients.patient_id"), index=True)
    hour_from_admission: Mapped[int] = mapped_column(Integer)
    wbc_count: Mapped[float] = mapped_column(Float)
    lactate: Mapped[float] = mapped_column(Float)
    creatinine: Mapped[float] = mapped_column(Float)
    crp_level: Mapped[float] = mapped_column(Float)
    hemoglobin: Mapped[float] = mapped_column(Float)

    patient = relationship("PatientRecord", back_populates="labs")


class Resource(Base):
    """Current operational capacity ('now' state used by the optimizer)."""
    __tablename__ = "resources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    category: Mapped[str] = mapped_column(String(40))  # bed|staffing|theatre|ed
    total: Mapped[int] = mapped_column(Integer)
    occupied: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime,
                                                 default=datetime.utcnow)


class DemandHistory(Base):
    """Observed daily demand per series (seeded from the ops dataset)."""
    __tablename__ = "demand_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    icu_demand: Mapped[int] = mapped_column(Integer)
    emergency_demand: Mapped[int] = mapped_column(Integer)
    general_demand: Mapped[int] = mapped_column(Integer)
    bed_occupancy: Mapped[float] = mapped_column(Float)
    staff_count: Mapped[float] = mapped_column(Float)
    flu_cases: Mapped[int] = mapped_column(Integer)


class ForecastRecord(Base):
    """A saved forecast run (one row per horizon per series)."""
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime,
                                                 default=datetime.utcnow)
    origin_date: Mapped[str] = mapped_column(String(10), index=True)
    horizon_hours: Mapped[int] = mapped_column(Integer)
    series: Mapped[str] = mapped_column(String(30))  # icu|emergency|general_beds
    predicted_value: Mapped[float] = mapped_column(Float)
    current_value: Mapped[float] = mapped_column(Float)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ScenarioRecord(Base):
    """What-if scenario definition + its projected outcome."""
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime,
                                                 default=datetime.utcnow)
    name: Mapped[str] = mapped_column(String(80))
    changes: Mapped[dict] = mapped_column(JSON)
    horizon_hours: Mapped[int] = mapped_column(Integer, default=48)
    projected_pressure: Mapped[dict] = mapped_column(JSON)
    shortages: Mapped[list] = mapped_column(JSON)


class OptimizationResult(Base):
    """OR-Tools solver output, recommendations and explanation for a run."""
    __tablename__ = "optimization_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime,
                                                 default=datetime.utcnow)
    scenario_id: Mapped[int | None] = mapped_column(
        ForeignKey("scenarios.id"), nullable=True)
    horizon_hours: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    allocation: Mapped[dict] = mapped_column(JSON)
    shortages: Mapped[list] = mapped_column(JSON)
    reserved_capacity: Mapped[dict] = mapped_column(JSON)
    recommendations: Mapped[list] = mapped_column(JSON)
    explanation: Mapped[list] = mapped_column(JSON)
    objective_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scenario_label: Mapped[str | None] = mapped_column(String(80), nullable=True)
    scenario_changes: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    projected_pressure: Mapped[dict | None] = mapped_column(JSON, nullable=True)
