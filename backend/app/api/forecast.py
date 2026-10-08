"""Forecast endpoints (24/48/72h model output)."""
from fastapi import APIRouter, Depends, HTTPException, Query

from ..security import require_auth
from ..services import forecast_service

router = APIRouter(prefix="/api/forecast", tags=["forecast"],
                   dependencies=[Depends(require_auth)])


@router.get("")
def full_forecast(horizon: int = Query(72, ge=24, le=72)):
    """Forecast for all series up to the requested horizon."""
    try:
        f = forecast_service.forecast(horizon)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    metrics = forecast_service.load_models()["metrics"]
    return {**f,
            "series_names": {"icu": "ICU demand", "emergency": "Emergency demand",
                             "general": "General bed demand"},
            "model_metrics": {k: {"test": v["test"],
                                  "test_by_horizon": v["test_by_horizon"],
                                  "algorithm": v["algorithm"]}
                              for k, v in metrics.items()},
            "units": "demand index (dataset units); see /api/dashboard for "
                     "hospital-unit projections",
            "data_note": "Synthetic/simulated hospital operations data."}


def _series(key: str, horizon: int):
    f = forecast_service.forecast(horizon)
    return {"series": key, "forecast_horizon": horizon,
            "timestamps": f["timestamps"], "timestamps_iso": f["timestamps_iso"],
            "values": f["series"][key],
            "current": f["series"][key][0]}


@router.get("/icu")
def icu(horizon: int = Query(72, ge=24, le=72)):
    return _series("icu", horizon)


@router.get("/emergency")
def emergency(horizon: int = Query(72, ge=24, le=72)):
    return _series("emergency", horizon)


@router.get("/beds")
def beds(horizon: int = Query(72, ge=24, le=72)):
    return _series("general_beds", horizon)
