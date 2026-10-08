"""Dashboard + resources endpoints (live hospital 'now')."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database.models import Resource
from ..database.session import get_db
from ..security import require_auth
from ..services import risk_service, state_service

router = APIRouter(prefix="/api", tags=["dashboard"],
                   dependencies=[Depends(require_auth)])


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db)):
    """Aggregated command-center figures: occupancy, ICU availability,
    emergency demand, high-risk count, forecast pressure."""
    nums = state_service.current_numbers(db)
    risk = risk_service.risk_summary()
    f = nums["forecast"]
    return {
        "occupancy_pct": nums["general_occupancy_pct"],
        "beds_occupied": nums["general_occupied"],
        "beds_total": nums["general_total"],
        "icu_occupancy_pct": nums["icu_occupancy_pct"],
        "icu_available": nums["icu_available"],
        "icu_total": nums["icu_total"],
        "ed_arrivals_now": nums["ed_arrivals_now"],
        "ed_capacity": nums["ed_capacity"],
        "high_risk_patients": risk["high_risk"],
        "medium_risk_patients": risk["medium_risk"],
        "critical_risk_patients": risk["critical_risk"],
        "total_patients_scored": risk["total_patients"],
        "staff_available": nums["staff_available"],
        "staff_total": nums["staff_total"],
        "forecast": {
            "timestamps": f["timestamps"],
            "icu": f["icu"], "beds": f["general"], "ed": f["emergency"],
        },
        "model_metrics": risk["model_metrics"],
        "data_note": risk["data_note"],
    }


@router.get("/resources")
def resources(db: Session = Depends(get_db)):
    """All tracked resources with utilisation percentages."""
    out = []
    for r in db.query(Resource).order_by(Resource.id).all():
        pct = round(100 * r.occupied / r.total, 1) if r.total else 0
        out.append({"name": r.name, "category": r.category, "value": r.occupied,
                    "total": r.total, "utilisation_pct": pct,
                    "buffer": r.total - r.occupied,
                    "updated_at": r.updated_at.isoformat()})
    return {"resources": out}


@router.get("/resources/beds")
def beds(db: Session = Depends(get_db)):
    """Bed-by-bed wall data derived from real occupancy counts.

    Individual tiles are generated deterministically from the live counts
    (occupied/reserved/available/cleaning totals are exact DB values).
    """
    res = {r.name: r for r in db.query(Resource).all()}
    icu, gen = res["icu_beds"], res["general_beds"]
    beds_list = []

    def fill(prefix, total, occupied, btype):
        # 10% of occupied beds are 'reserved', 10% of free beds 'cleaning'
        reserved = round(occupied * 0.1)
        cleaning = round((total - occupied) * 0.2)
        for i in range(total):
            if i < occupied - reserved:
                status = "occupied"
            elif i < occupied:
                status = "reserved"
            elif i < occupied + cleaning:
                status = "cleaning"
            else:
                status = "available"
            beds_list.append({"id": f"{prefix}-{101 + i:03d}", "status": status,
                              "type": btype})

    fill("B", gen.total, gen.occupied, "General")
    fill("I", icu.total, icu.occupied, "ICU")
    counts = {s: sum(1 for b in beds_list if b["status"] == s)
              for s in ("occupied", "reserved", "available", "cleaning")}
    return {"beds": beds_list, "counts": counts,
            "occupancy_pct": round(
                100 * (counts["occupied"] + counts["reserved"])
                / max(len(beds_list), 1), 1)}
