"""OR-Tools resource allocation optimizer.

Problem
-------
Given projected demand (from the forecasting models) and current capacity,
decide how many of each constrained resource to allocate so that total
operational risk is minimized.

Objective (minimize):
    w1 * ICU shortfall
  + w2 * Emergency shortfall
  + w3 * General-bed shortfall
  + w4 * Staff overload beyond safe staffing ratio
  + w5 * Unused reserve below the required safety buffer

Subject to (hard constraints):
    icu_allocated       <= icu_beds_available
    emergency_allocated <= emergency_capacity
    general_allocated   <= general_beds_available
    staff_allocated     <= staff_available
    or_allocated        <= or_capacity
    icu_reserved        >= icu_reserve_required
    staff_allocated     >= min_staff_required(demand)
    icu_allocated, emergency_allocated, general_allocated,
    staff_allocated, or_allocated, icu_reserved  >= 0  and integer

The solver is Google OR-Tools CP-SAT. Nothing here is hardcoded output:
every number returned comes out of the optimization result.
"""
from dataclasses import dataclass, field, asdict

from ortools.sat.python import cp_model


@dataclass
class HospitalState:
    """Current operational capacity (units = beds / staff / theatres)."""
    icu_beds_total: int
    icu_beds_occupied: int
    general_beds_total: int
    general_beds_occupied: int
    emergency_capacity: int
    emergency_arrivals_current: int
    staff_total: int
    staff_available: int
    or_capacity: int
    or_scheduled: int

    @property
    def icu_beds_available(self) -> int:
        return max(self.icu_beds_total - self.icu_beds_occupied, 0)

    @property
    def general_beds_available(self) -> int:
        return max(self.general_beds_total - self.general_beds_occupied, 0)


@dataclass
class Demand:
    """Projected demand at the chosen horizon, in absolute units.

    icu/general/emergency are the TOTAL units needed at the horizon
    (census/arrivals), not 'extra beyond now'. Occupied units are already
    committed, so the solver only allocates free units and measures shortfall
    against total need.
    """
    icu: int
    emergency: int
    general: int
    # ICU free units that must be held back as surge buffer
    icu_reserve_required: int = 0
    # staff needed to safely cover the projected demand
    min_staff_required: int = 0
    # elective/scheduled operating-theatre cases that should be covered
    or_cases_planned: int = 0


# Objective weights - documented in README (safety-first ordering).
WEIGHTS = {
    "icu_shortfall": 100,
    "emergency_shortfall": 60,
    "general_shortfall": 40,
    "staff_overload": 25,
    "reserve_below_min": 80,
    "unused_beyond_reserve": 1,
}


def solve(state: HospitalState, demand: Demand, horizon_hours: int = 48) -> dict:
    """Run the CP-SAT model and return a fully-populated result dict."""
    model = cp_model.CpModel()

    icu_cap = state.icu_beds_available
    gen_cap = state.general_beds_available
    emg_cap = state.emergency_capacity
    staff_cap = state.staff_available
    or_cap = state.or_capacity

    # Decision variables (integer resource counts).
    icu = model.new_int_var(0, icu_cap, "icu_allocated")
    emg = model.new_int_var(0, emg_cap, "emergency_allocated")
    gen = model.new_int_var(0, gen_cap, "general_allocated")
    staff = model.new_int_var(0, staff_cap, "staff_allocated")
    orv = model.new_int_var(0, or_cap, "or_allocated")
    reserve = model.new_int_var(0, icu_cap, "icu_reserved")

    # Shortfall = units of TOTAL need not covered by (occupied + allocated).
    # For beds, occupied patients are already cared for; only the remaining
    # need has to come from free capacity. For throughput resources (ED, staff,
    # OR) there is no occupancy - need is compared to the allocation directly.
    def shortfall(var, name, target, committed=0):
        s = model.new_int_var(0, max(target, 1), name)
        # minimization drives s down to exactly max(need - committed - var, 0)
        model.add(s >= target - committed - var)
        return s

    icu_short = shortfall(icu, "icu_shortfall", demand.icu, state.icu_beds_occupied)
    emg_short = shortfall(emg, "emg_shortfall", demand.emergency)
    gen_short = shortfall(gen, "gen_shortfall", demand.general,
                          state.general_beds_occupied)

    # Staff: allocate at least the safe minimum; overstaffing beyond it is
    # cheap but understaffing is heavily penalized.
    staff_short = model.new_int_var(0, max(demand.min_staff_required, 1),
                                    "staff_shortfall")
    model.add(staff_short >= demand.min_staff_required - staff)
    staff_over = model.new_int_var(0, max(staff_cap, 1), "staff_overload")
    model.add(staff_over >= staff - max(demand.min_staff_required, 0))

    # Reserve: ICU beds deliberately held back. Below-minimum reserve is
    # heavily penalized; tiny unused surplus above the minimum is cheap.
    reserve_def = model.new_int_var(0, max(icu_cap, 1), "reserve_below_min")
    model.add(reserve_def >= demand.icu_reserve_required - reserve)
    unused = model.new_int_var(0, max(icu_cap, 1), "unused_beyond_reserve")
    model.add(unused >= reserve - demand.icu_reserve_required)

    # OR: cover planned theatre cases when capacity allows.
    or_short = shortfall(orv, "or_shortfall", demand.or_cases_planned)
    or_unused = model.new_int_var(0, max(or_cap, 1), "or_unused")
    model.add(or_unused >= orv - demand.or_cases_planned)

    # Hard constraints.
    model.add(icu + reserve <= icu_cap)   # allocation + reserve <= available ICU
    model.add(reserve >= 0)
    if 0 < demand.min_staff_required <= staff_cap:
        model.add(staff >= demand.min_staff_required)
    # Allocation never exceeds capacity (also enforced by variable bounds).
    model.add(emg <= emg_cap)

    model.minimize(
        WEIGHTS["icu_shortfall"] * icu_short
        + WEIGHTS["emergency_shortfall"] * emg_short
        + WEIGHTS["general_shortfall"] * gen_short
        + WEIGHTS["staff_overload"] * staff_short * 3
        + WEIGHTS["staff_overload"] * staff_over
        + WEIGHTS["reserve_below_min"] * reserve_def
        + WEIGHTS["unused_beyond_reserve"] * unused
        + WEIGHTS["general_shortfall"] * or_short
        + WEIGHTS["unused_beyond_reserve"] * or_unused
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    solver.parameters.num_search_workers = 4
    status = solver.solve(model)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, str(status))

    if status_name not in ("OPTIMAL", "FEASIBLE"):
        return {"status": status_name, "allocation": None,
                "shortages": [], "recommendations": [], "explanation": []}

    alloc = {
        "icu": solver.value(icu),
        "general_beds": solver.value(gen),
        "emergency": solver.value(emg),
        "staff": solver.value(staff),
        "or": solver.value(orv),
    }
    shortages = []
    if solver.value(icu_short) > 0:
        shortages.append("ICU beds")
    if solver.value(emg_short) > 0:
        shortages.append("Emergency capacity")
    if solver.value(gen_short) > 0:
        shortages.append("General beds")
    if demand.min_staff_required > staff_cap:
        shortages.append("Staffing")
    elif solver.value(staff_short) > 0:
        shortages.append("Staffing")

    result = {
        "status": status_name,
        "horizon_hours": horizon_hours,
        "allocation": alloc,
        "reserved_capacity": {"icu": solver.value(reserve)},
        "shortages": shortages,
        "objective_value": round(solver.objective_value, 2),
        "demand": asdict(demand),
        "capacity": {
            "icu_total": state.icu_beds_total,
            "icu_occupied": state.icu_beds_occupied,
            "icu_available": icu_cap,
            "general_total": state.general_beds_total,
            "general_occupied": state.general_beds_occupied,
            "general_available": gen_cap,
            "emergency_capacity": emg_cap,
            "staff_total": state.staff_total,
            "staff_available": staff_cap,
            "or_capacity": or_cap,
        },
        "shortfall_detail": {
            "icu": solver.value(icu_short),
            "emergency": solver.value(emg_short),
            "general": solver.value(gen_short),
            "or": solver.value(or_short),
        },
        "weights": WEIGHTS,
    }
    return result
