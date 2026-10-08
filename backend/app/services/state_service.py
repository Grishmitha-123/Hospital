"""Current-state service: reads live capacity from PostgreSQL and converts
model forecasts (dataset demand units) into this hospital's bed/staff units.

Scaling rule (documented, no fabricated numbers):
  The ops dataset reports demand as counts for a reference hospital. The
  demonstration hospital has configured capacity (CAPACITY in seed.py), so a
  projected census is derived as

      census(h) = current_census * demand(h) / demand(now)

  i.e. the model's own growth ratio is applied to the observed census.
  Emergency demand is absolute arrivals/day and maps 1:1 onto ED throughput
  capacity in arrivals/day.
"""
from __future__ import annotations

import math

from sqlalchemy.orm import Session

from ..database.models import Resource
from ..optimization.solver import HospitalState, Demand
from . import forecast_service

# Demo-hospital safe staffing ratios (documented in README/model report).
RATIO_ICU_NURSES = 2.0      # nurses per ICU bed
RATIO_GENERAL_NURSES = 4.0  # beds per nurse on general wards
RATIO_ED_PER_ARRIVAL = 10.0  # arrivals per ED staff member
ICU_RESERVE_PCT = 0.05      # share of ICU total kept free as surge buffer


def _resources(db: Session) -> dict:
    return {r.name: r for r in db.query(Resource).all()}


def get_state(db: Session) -> HospitalState:
    res = _resources(db)
    return HospitalState(
        icu_beds_total=res["icu_beds"].total,
        icu_beds_occupied=res["icu_beds"].occupied,
        general_beds_total=res["general_beds"].total,
        general_beds_occupied=res["general_beds"].occupied,
        emergency_capacity=res["emergency_capacity"].total,
        emergency_arrivals_current=res["emergency_capacity"].occupied,
        staff_total=res["staff"].total,
        staff_available=res["staff"].occupied,
        or_capacity=res["or_theatres"].total,
        or_scheduled=res["or_theatres"].occupied,
    )


def required_staff(icu_need: int, general_need: int, emergency_need: int) -> int:
    """Minimum staff to safely cover projected demand (safe ratios above)."""
    return int(math.ceil(
        icu_need / RATIO_ICU_NURSES
        + general_need / RATIO_GENERAL_NURSES
        + emergency_need / RATIO_ED_PER_ARRIVAL))


def get_demand(db: Session, horizon_hours: int = 48) -> Demand:
    """Forecast demand at the horizon, expressed in this hospital's units."""
    state = get_state(db)
    f = forecast_service.forecast(max(horizon_hours, 24))
    idx = {24: 1, 48: 2, 72: 3}[horizon_hours]
    icu_now, icu_h = f["icu"][0], f["icu"][idx]
    gen_now, gen_h = f["general"][0], f["general"][idx]
    emg_h = f["emergency"][idx]

    icu_need = int(round(state.icu_beds_occupied * icu_h / max(icu_now, 1)))
    gen_need = int(round(state.general_beds_occupied * gen_h / max(gen_now, 1)))
    # Never claim need below what is already occupied, nor absurdly above.
    icu_need = max(icu_need, state.icu_beds_occupied)
    gen_need = max(gen_need, state.general_beds_occupied)

    return Demand(
        icu=icu_need,
        emergency=int(emg_h),
        general=gen_need,
        icu_reserve_required=max(1, int(round(
            ICU_RESERVE_PCT * state.icu_beds_total))),
        min_staff_required=required_staff(icu_need, gen_need, int(emg_h)),
        or_cases_planned=state.or_scheduled,
    )


def current_numbers(db: Session) -> dict:
    """Dashboard 'now' figures (all sourced from DB + models)."""
    state = get_state(db)
    f = forecast_service.forecast(72)
    return {
        "general_occupied": state.general_beds_occupied,
        "general_total": state.general_beds_total,
        "general_occupancy_pct": round(
            100 * state.general_beds_occupied / state.general_beds_total, 1),
        "icu_occupied": state.icu_beds_occupied,
        "icu_total": state.icu_beds_total,
        "icu_available": state.icu_beds_available,
        "icu_occupancy_pct": round(
            100 * state.icu_beds_occupied / state.icu_beds_total, 1),
        "ed_arrivals_now": state.emergency_arrivals_current,
        "ed_capacity": state.emergency_capacity,
        "staff_available": state.staff_available,
        "staff_total": state.staff_total,
        "or_capacity": state.or_capacity,
        "or_scheduled": state.or_scheduled,
        "forecast": f,
    }
