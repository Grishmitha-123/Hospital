"""Backend tests: auth, dashboard, risk, forecast, simulation, optimization.

Requires the seeded PostgreSQL database (python -m app.database.seed) and the
trained model artifacts in backend/models/.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.optimization.solver import Demand, HospitalState, solve

client = TestClient(app)


@pytest.fixture(scope="module")
def token() -> str:
    r = client.post("/api/auth/login",
                    json={"email": "admin@medicare.ai", "password": "medicare123"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_login_rejects_bad_password():
    r = client.post("/api/auth/login",
                    json={"email": "admin@medicare.ai", "password": "wrong"})
    assert r.status_code == 401


def test_protected_endpoints_require_auth():
    for path in ("/api/dashboard", "/api/patients", "/api/forecast",
                 "/api/risk/summary", "/api/optimization/latest"):
        assert client.get(path).status_code == 401, path


def test_dashboard(auth):
    d = client.get("/api/dashboard", headers=auth).json()
    assert d["beds_total"] > 0
    assert 0 <= d["occupancy_pct"] <= 100
    assert d["high_risk_patients"] >= 0
    assert len(d["forecast"]["icu"]) == 4  # now + 24/48/72h


def test_patient_risk_endpoint(auth):
    lst = client.get("/api/patients?limit=5", headers=auth).json()["patients"]
    assert len(lst) == 5
    pid = lst[0]["patient_id"]
    p = client.get(f"/api/patients/{pid}/risk", headers=auth).json()
    assert p["patient_id"] == pid
    assert 0.0 <= p["risk_probability"] <= 1.0
    assert p["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(p["factors"], list) and p["factors"]


def test_patient_not_found(auth):
    assert client.get("/api/patients/999999/risk", headers=auth).status_code == 404


def test_risk_summary(auth):
    s = client.get("/api/risk/summary", headers=auth).json()
    assert s["total_patients"] == 10000
    assert s["high_risk"] + s["medium_risk"] + s["low_risk"] == s["total_patients"]
    assert s["model_metrics"]["roc_auc"] > 0.5


def test_forecast_horizons(auth):
    for h in (24, 48, 72):
        f = client.get(f"/api/forecast?horizon={h}", headers=auth).json()
        assert f["forecast_horizon"] == h
        assert len(f["timestamps"]) == h // 24 + 1
        # each horizon must use its own model -> values may not all be equal
        assert all(v > 0 for v in f["icu"])
    bad = client.get("/api/forecast?horizon=12", headers=auth)
    assert bad.status_code == 422


def test_forecast_series_endpoints(auth):
    for path in ("/api/forecast/icu", "/api/forecast/emergency", "/api/forecast/beds"):
        d = client.get(path, headers=auth).json()
        assert len(d["values"]) == 4


def test_simulation_run(auth):
    r = client.post("/api/simulation/run", headers=auth,
                    json={"preset": "surge40", "horizon_hours": 48})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["scenario"]["emergency_arrivals_change"] == 40
    assert "emergency" in d["projected_pressure"]
    assert d["optimization"]["status"] in ("OPTIMAL", "FEASIBLE")
    assert d["recommendations"]
    assert d["explanation"]


def test_optimization_run_and_latest(auth):
    r = client.post("/api/optimization/run", headers=auth,
                    json={"horizon_hours": 48})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] in ("OPTIMAL", "FEASIBLE")
    latest = client.get("/api/optimization/latest", headers=auth).json()
    assert latest["allocation"] is not None


# ---------------- optimizer unit tests (no API) ----------------

def _state(**kw):
    base = dict(icu_beds_total=30, icu_beds_occupied=20,
                general_beds_total=100, general_beds_occupied=60,
                emergency_capacity=120, emergency_arrivals_current=80,
                staff_total=80, staff_available=60,
                or_capacity=10, or_scheduled=6)
    base.update(kw)
    return HospitalState(**base)


def test_optimizer_feasible_when_capacity_suffices():
    st = _state()
    dm = Demand(icu=25, emergency=100, general=80,
                icu_reserve_required=2, min_staff_required=50, or_cases_planned=6)
    r = solve(st, dm, 48)
    assert r["status"] == "OPTIMAL"
    # hard constraints respected
    assert r["allocation"]["icu"] + r["reserved_capacity"]["icu"] <= st.icu_beds_available
    assert r["allocation"]["emergency"] <= st.emergency_capacity
    assert r["allocation"]["staff"] <= st.staff_available
    assert r["allocation"]["or"] <= st.or_capacity
    assert r["shortages"] == []


def test_optimizer_flags_shortage_when_demand_exceeds_capacity():
    st = _state()
    dm = Demand(icu=40, emergency=300, general=200,
                icu_reserve_required=2, min_staff_required=500, or_cases_planned=6)
    r = solve(st, dm, 72)
    assert r["status"] in ("OPTIMAL", "FEASIBLE")
    assert set(r["shortages"]) >= {"ICU beds", "Emergency capacity", "General beds"}
    # allocation never exceeds capacity even under extreme demand
    assert r["allocation"]["icu"] <= st.icu_beds_available
    assert r["allocation"]["emergency"] <= st.emergency_capacity
    assert r["allocation"]["staff"] <= st.staff_available
