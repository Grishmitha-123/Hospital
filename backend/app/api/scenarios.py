"""Simulation + optimization endpoints (PREDICT -> SIMULATE -> OPTIMIZE -> EXPLAIN)."""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database.models import OptimizationResult, ScenarioRecord
from ..database.session import get_db
from ..optimization.explanation import explain, recommendations
from ..optimization.solver import solve
from ..security import require_auth
from ..services import state_service
from ..simulation.engine import CHANGE_KEYS, PRESETS, run_simulation

router = APIRouter(prefix="/api", tags=["simulation-optimization"],
                   dependencies=[Depends(require_auth)])


class ScenarioIn(BaseModel):
    preset: str | None = Field(None, description="Key of a predefined scenario")
    changes: dict = Field(default_factory=dict,
                          description="Arbitrary deltas, e.g. "
                                      "{'emergency_arrivals_change': 20}")
    horizon_hours: int = Field(48, ge=24, le=72)


@router.get("/simulation/presets")
def presets():
    """Scenario buttons available in the UI + meaning of each change key."""
    return {"presets": [{"id": k, "label": v["label"], "changes": v["changes"]}
                        for k, v in PRESETS.items()],
            "change_keys": CHANGE_KEYS}


@router.post("/simulation/run")
def run(body: ScenarioIn, db: Session = Depends(get_db)):
    """Apply a what-if scenario to the live state + forecast, then optimize."""
    changes = dict(body.changes)
    if body.preset:
        changes = {**PRESETS.get(body.preset, {}).get("changes", {}), **changes}
        changes["_preset"] = body.preset or "custom"
    state = state_service.get_state(db)
    demand = state_service.get_demand(db, body.horizon_hours)
    result = run_simulation(state, demand, changes, body.horizon_hours)

    rec = ScenarioRecord(
        name=result["scenario_label"],
        changes=result["scenario"],
        horizon_hours=body.horizon_hours,
        projected_pressure=result["projected_pressure"],
        shortages=result["shortages"],
    )
    db.add(rec)
    db.flush()
    opt = result["optimization"]
    db.add(OptimizationResult(
        scenario_id=rec.id,
        horizon_hours=body.horizon_hours,
        status=opt["status"],
        allocation=opt["allocation"],
        shortages=opt["shortages"],
        reserved_capacity=opt["reserved_capacity"],
        recommendations=result["recommendations"],
        explanation=result["explanation"],
        objective_value=opt.get("objective_value"),
        raw=opt,
        scenario_label=result["scenario_label"],
        scenario_changes=result["scenario"],
        projected_pressure=result["projected_pressure"],
    ))
    db.commit()
    return {**result, "scenario_id": rec.id}


@router.post("/optimization/run")
def optimization_run(body: ScenarioIn, db: Session = Depends(get_db)):
    """Run OR-Tools directly against current state + forecast demand."""
    changes = dict(body.changes)
    if body.preset:
        changes = {**PRESETS.get(body.preset, {}).get("changes", {}), **changes}
    state = state_service.get_state(db)
    demand = state_service.get_demand(db, body.horizon_hours)
    from ..simulation.engine import apply_scenario
    s2, d2, applied = apply_scenario(state, demand, changes)
    opt = solve(s2, d2, horizon_hours=body.horizon_hours)
    rec = OptimizationResult(
        scenario_id=None,
        horizon_hours=body.horizon_hours,
        status=opt["status"],
        allocation=opt.get("allocation") or {},
        shortages=opt.get("shortages", []),
        reserved_capacity=opt.get("reserved_capacity", {}),
        recommendations=recommendations(opt, s2, d2),
        explanation=explain(opt, s2, d2, scenario=applied),
        objective_value=opt.get("objective_value"),
        raw=opt,
        scenario_label="Direct optimization",
        scenario_changes=applied,
    )
    db.add(rec)
    db.commit()
    return {**opt, "recommendations": rec.recommendations,
            "explanation": rec.explanation}


@router.get("/optimization/latest")
def optimization_latest(db: Session = Depends(get_db)):
    """Most recent optimization run (as stored in PostgreSQL)."""
    row = (db.query(OptimizationResult)
           .order_by(OptimizationResult.id.desc()).first())
    if row is None:
        # Nothing run yet: compute once from current state so the page works.
        state = state_service.get_state(db)
        demand = state_service.get_demand(db, 48)
        opt = solve(state, demand, horizon_hours=48)
        return {**opt,
                "recommendations": recommendations(opt, state, demand),
                "explanation": explain(opt, state, demand),
                "created_at": None, "id": None}
    raw = row.raw or {}
    return {"id": row.id, "created_at": row.created_at.isoformat(),
            "horizon_hours": row.horizon_hours, "status": row.status,
            "allocation": row.allocation, "shortages": row.shortages,
            "reserved_capacity": row.reserved_capacity,
            "recommendations": row.recommendations,
            "explanation": row.explanation,
            "objective_value": row.objective_value,
            "scenario_label": row.scenario_label,
            "scenario_changes": row.scenario_changes,
            "projected_pressure": row.projected_pressure,
            "demand": raw.get("demand"),
            "capacity": raw.get("capacity"),
            "shortfall_detail": raw.get("shortfall_detail"),
            "weights": raw.get("weights")}
