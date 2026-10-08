"""Patient risk endpoints (real model inference)."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..security import require_auth
from ..services import risk_service

router = APIRouter(prefix="/api", tags=["patients"],
                   dependencies=[Depends(require_auth)])


@router.get("/patients")
def patients(limit: int = Query(25, ge=1, le=200),
             level: str | None = Query(None, pattern="^(LOW|MEDIUM|HIGH)$"),
             db: Session = Depends(get_db)):
    """Highest-risk patients first (batch-scored by the trained model)."""
    return {"patients": risk_service.list_patients(top_n=limit, min_level=level),
            "prediction_window": "12h from admission"}


@router.get("/patients/{patient_id}/risk")
def patient_risk(patient_id: int):
    """Full risk record for one patient: probability, level, factors."""
    rec = risk_service.score_patient(patient_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return rec


@router.get("/risk/summary")
def risk_summary():
    """Cohort-level risk distribution + model evaluation metrics."""
    return risk_service.risk_summary()
