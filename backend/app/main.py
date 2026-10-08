"""MediCare AI FastAPI application.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import auth, dashboard, forecast, patients, scenarios
from .config import CORS_ORIGINS
from .services import forecast_service, risk_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm model caches at startup so the first request is fast.
    forecast_service.load_models()
    risk_service._load()
    yield


app = FastAPI(
    title="MediCare AI API",
    description=(
        "Hospital resource optimization platform: patient risk, demand "
        "forecasting, what-if simulation, OR-Tools optimization and "
        "explanations. Data is synthetic/simulated for demonstration."),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(patients.router)
app.include_router(forecast.router)
app.include_router(scenarios.router)


@app.get("/api/health")
def health():
    """Unauthenticated liveness probe for the frontend."""
    return {"status": "ok", "service": "medicare-ai", "version": "1.0.0"}
