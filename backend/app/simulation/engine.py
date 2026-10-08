"""What-if simulation engine.

Pipeline: current state -> forecast demand -> apply scenario deltas ->
projected pressure + shortages -> OR-Tools optimizer -> recommendation.

Nothing here is hardcoded output: pressure, shortages and the
recommendation all come from arithmetic on the forecast/state and from the
CP-SAT solver.
"""
from __future__ import annotations

from dataclasses import asdict

from ..optimization.explanation import explain, recommendations
from ..optimization.solver import Demand, HospitalState, solve

# Predefined scenarios shown as buttons in the existing frontend.
PRESETS = {
    "base": {"label": "Current plan", "changes": {}},
    "surge20": {"label": "Emergency arrivals +20%",
                "changes": {"emergency_arrivals_change": 20}},
    "surge40": {"label": "Emergency arrivals +40%",
                "changes": {"emergency_arrivals_change": 40}},
    "icu2": {"label": "ICU beds -2", "changes": {"icu_capacity_change": -2}},
    "nurses3": {"label": "Nurses -3", "changes": {"staff_change": -4}},
    "surgery15": {"label": "Surgical demand +15%",
                  "changes": {"or_demand_change": 15}},
    "flu10": {"label": "Flu surge +10%", "changes": {"flu_surge_pct": 10}},
    "staff10": {"label": "Staff shortage -10%",
                "changes": {"staff_change": -10}},
    "icu5": {"label": "ICU capacity -5", "changes": {"icu_capacity_change": -5}},
}

# Which scenario fields affect what (documented mapping).
CHANGE_KEYS = {
    "emergency_arrivals_change": "% change to emergency demand",
    "admission_surge_change": "% change to general-bed demand",
    "flu_surge_pct": "% increase in flu-driven demand (applies to all units)",
    "staff_change": "% change to available staff",
    "icu_capacity_change": "absolute change to ICU bed capacity",
    "bed_capacity_change": "absolute change to general bed capacity",
    "or_demand_change": "% change to planned OR cases",
    "or_availability_change": "absolute change to OR capacity",
}


def apply_scenario(state: HospitalState, demand: Demand, changes: dict):
    """Return (modified_state, modified_demand, applied_changes).

    The special key `_preset` only selects a label; it never enters the
    numeric scenario mapping.
    """
    c = {k: float(v) for k, v in (changes or {}).items()
         if v and k in CHANGE_KEYS}
    flu = 1.0 + c.get("flu_surge_pct", 0.0) / 100.0
    emg = 1.0 + c.get("emergency_arrivals_change", 0.0) / 100.0
    adm = 1.0 + c.get("admission_surge_change", 0.0) / 100.0

    new_state = HospitalState(**asdict(state))
    new_demand = Demand(**asdict(demand))

    # Capacity-side changes (absolute units).
    if "icu_capacity_change" in c:
        delta = int(c["icu_capacity_change"])
        new_state.icu_beds_total = max(new_state.icu_beds_total + delta, 0)
    if "bed_capacity_change" in c:
        delta = int(c["bed_capacity_change"])
        new_state.general_beds_total = max(new_state.general_beds_total + delta, 0)
    if "or_availability_change" in c:
        delta = int(c["or_availability_change"])
        new_state.or_capacity = max(new_state.or_capacity + delta, 0)
    if "staff_change" in c:
        factor = 1.0 + c["staff_change"] / 100.0
        new_state.staff_available = max(int(round(new_state.staff_available * factor)), 0)

    # Demand-side changes.
    new_demand.emergency = max(int(round(new_demand.emergency * emg * flu)), 0)
    new_demand.general = max(int(round(new_demand.general * adm * flu)), 0)
    new_demand.icu = max(int(round(new_demand.icu * flu)), 0)
    if "or_demand_change" in c:
        factor = 1.0 + c["or_demand_change"] / 100.0
        new_demand.or_cases_planned = max(
            int(round(new_demand.or_cases_planned * factor)), 0)
    # Safe staffing scales with demand pressure.
    new_demand.min_staff_required = max(
        int(round(new_demand.min_staff_required * max(emg, adm, flu))), 0)

    applied = {k: v for k, v in c.items()}
    return new_state, new_demand, applied


def pressure(state: HospitalState, demand: Demand) -> dict:
    """Projected utilisation % for each resource (demand / capacity)."""
    def pct(d, c):
        return int(round(100 * d / c)) if c > 0 else 100
    return {
        "icu": pct(demand.icu, state.icu_beds_total),
        "emergency": pct(demand.emergency, state.emergency_capacity),
        "general_beds": pct(demand.general, state.general_beds_total),
        "staff": pct(demand.min_staff_required, state.staff_total),
        "or": pct(demand.or_cases_planned, state.or_capacity),
    }


def shortages(press: dict) -> list[str]:
    names = {"icu": "ICU beds", "emergency": "Emergency capacity",
             "general_beds": "General beds", "staff": "Staffing",
             "or": "OR capacity"}
    return [names[k] for k, v in press.items() if v > 100]


def run_simulation(state: HospitalState, demand: Demand, changes: dict,
                   horizon_hours: int = 48) -> dict:
    """Full PREDICT -> SIMULATE -> OPTIMIZE -> EXPLAIN chain for one scenario."""
    new_state, new_demand, applied = apply_scenario(state, demand, changes)
    press = pressure(new_state, new_demand)
    short = shortages(press)
    opt = solve(new_state, new_demand, horizon_hours=horizon_hours)
    return {
        "scenario": applied,
        "scenario_label": PRESETS.get(
            (changes or {}).get("_preset", "base"),
            {"label": "Custom scenario"})["label"],
        "horizon_hours": horizon_hours,
        "projected_pressure": press,
        "shortages": short,
        "optimization": opt,
        "recommendations": recommendations(opt, new_state, new_demand),
        "explanation": explain(opt, new_state, new_demand, scenario=applied),
        "state_after": asdict(new_state),
        "demand_after": asdict(new_demand),
    }
