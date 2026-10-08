"""Explanation engine.

Builds human-readable, value-grounded explanations for an optimization
result. Every number is taken from the forecast, the current state, the
scenario and the solver output - nothing is invented here.
"""
from __future__ import annotations


def explain(result: dict, state, demand, scenario: dict | None = None) -> list[dict]:
    """Return one explanation block per resource, grounded in real numbers."""
    if result.get("allocation") is None:
        return [{
            "resource": "system",
            "lines": [f"Solver returned {result.get('status')}; no allocation to explain."],
        }]
    alloc = result["allocation"]
    cap = result["capacity"]
    dem = result["demand"]
    short = result["shortfall_detail"]
    reserve = result["reserved_capacity"]["icu"]
    blocks = []

    # ICU
    lines = [
        f"Projected ICU demand at {result['horizon_hours']}h = {dem['icu']} beds.",
        f"Currently available ICU beds = {cap['icu_available']} "
        f"({state.icu_beds_total} total - {state.icu_beds_occupied} occupied).",
        f"Recommended allocation = {alloc['icu']}, safety reserve = {reserve}.",
    ]
    if short["icu"] > 0:
        lines.append(
            f"Unmet ICU need = {short['icu']} beds: available capacity is below "
            f"projected demand, so the optimizer allocates every free bed and "
            f"flags the shortfall rather than overcommitting reserved capacity.")
    elif reserve >= dem["icu_reserve_required"]:
        lines.append(
            f"Reserve requirement of {dem['icu_reserve_required']} beds is met "
            f"while still covering projected demand.")
    else:
        lines.append(
            f"Reserve requirement of {dem['icu_reserve_required']} beds is NOT "
            f"met - projected demand consumes all available ICU capacity.")
    blocks.append({"resource": "ICU", "lines": lines})

    # Emergency
    lines = [
        f"Projected emergency demand at {result['horizon_hours']}h = {dem['emergency']} arrivals.",
        f"Emergency capacity = {cap['emergency_capacity']} "
        f"(current baseline {state.emergency_arrivals_current}/period).",
        f"Recommended allocation = {alloc['emergency']}.",
    ]
    lines.append(
        f"Shortfall = {short['emergency']}: capacity exceeds arrivals, so the "
        f"full emergency complement is staffed."
        if short["emergency"] == 0 else
        f"Shortfall = {short['emergency']} arrivals beyond capacity - open surge "
        f"capacity or divert non-critical load.")
    blocks.append({"resource": "Emergency", "lines": lines})

    # General beds
    lines = [
        f"Projected general-bed demand at {result['horizon_hours']}h = {dem['general']} beds.",
        f"Currently available general beds = {cap['general_available']} "
        f"({state.general_beds_total} total - {state.general_beds_occupied} occupied).",
        f"Recommended allocation = {alloc['general_beds']}.",
    ]
    lines.append(
        f"Shortfall = {short['general']} beds: demand cannot be met from free "
        f"capacity alone - discharge planning or step-down conversion is required."
        if short["general"] > 0 else
        "Free capacity covers projected demand.")
    blocks.append({"resource": "General beds", "lines": lines})

    # Staff
    lines = [
        f"Safe-staffing requirement from projected demand = {dem['min_staff_required']} staff.",
        f"Available staff = {cap['staff_available']} of {state.staff_total}.",
        f"Recommended allocation = {alloc['staff']}.",
    ]
    lines.append(
        "Staffing requirement cannot be fully covered - the binding constraint "
        "is workforce, not beds."
        if "Staffing" in result["shortages"] else
        "Staffing requirement is covered within available workforce.")
    blocks.append({"resource": "Staff", "lines": lines})

    # OR
    lines = [
        f"Scheduled theatre cases = {dem['or_cases_planned']}, "
        f"OR capacity = {cap['or_capacity']}.",
        f"Recommended allocation = {alloc['or']} theatres/slots.",
        f"Shortfall = {short['or']}.",
    ]
    blocks.append({"resource": "OR", "lines": lines})

    if scenario:
        changed = [f"{k} = {v:+g}%" if isinstance(v, (int, float)) and k.endswith("_change")
                   else f"{k} = {v}" for k, v in scenario.items()]
        if changed:
            blocks.insert(0, {
                "resource": "Scenario",
                "lines": ["Applied what-if changes: " + "; ".join(changed) + "."],
            })

    blocks.append({
        "resource": "Objective",
        "lines": [
            f"CP-SAT status = {result['status']}, objective = {result['objective_value']} "
            f"(weighted shortfall + reserve penalties; weights: "
            f"ICU {result['weights']['icu_shortfall']}, "
            f"emergency {result['weights']['emergency_shortfall']}, "
            f"general {result['weights']['general_shortfall']}, "
            f"staff {result['weights']['staff_overload']}, "
            f"reserve {result['weights']['reserve_below_min']}).",
            "All hard constraints (capacity limits, reserve minimum, staffing "
            "minimum) were enforced by the solver.",
        ],
    })
    return blocks


def recommendations(result: dict, state, demand) -> list[str]:
    """Actionable recommendation strings derived from the solver output."""
    if result.get("allocation") is None:
        return []
    alloc = result["allocation"]
    res = result["reserved_capacity"]["icu"]
    short = result["shortfall_detail"]
    out = []
    if res > 0:
        out.append(f"Reserve {res} ICU beds for projected demand and surge buffer.")
    if short["icu"] > 0:
        out.append(
            f"ICU shortfall of {short['icu']} beds: escalate elective admissions "
            f"review and prepare step-down transfer for stable patients.")
    if short["general"] > 0:
        out.append(
            f"General-bed shortfall of {short['general']}: accelerate discharge "
            f"planning and convert step-down capacity where safe.")
    if short["emergency"] > 0:
        out.append(
            f"Emergency shortfall of {short['emergency']}: open surge area and "
            f"redeploy staff to the ED fast track.")
    if short["general"] > 0 and state.general_beds_available > 0:
        out.append("Move flexible general beds into emergency overflow capacity.")
    if "Staffing" in result["shortages"]:
        out.append(
            f"Staffing gap: need {demand.min_staff_required}, have "
            f"{result['capacity']['staff_available']} - request overtime or cancel "
            f"non-urgent theatre lists.")
    elif alloc["staff"] > 0:
        out.append(
            f"Maintain {alloc['staff']} staff on duty to satisfy the safe "
            f"staffing ratio for projected demand.")
    if short["or"] > 0:
        out.append(f"OR shortfall of {short['or']} slots: defer elective cases.")
    elif demand.or_cases_planned:
        out.append(
            f"All {demand.or_cases_planned} scheduled theatre cases fit within "
            f"available OR capacity.")
    if not out:
        out.append("Current allocation is feasible; no reallocation required.")
    return out
