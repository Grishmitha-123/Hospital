# MediCare AI — Unified Hospital Resource Optimization Platform

**PREDICT → SIMULATE → OPTIMIZE → EXPLAIN**

A hospital operations decision-support system built on top of the existing
MediCare AI React frontend. It forecasts demand, scores patient deterioration
risk, simulates what-if disruptions, allocates constrained resources with
Google OR-Tools, and explains every recommendation using real numbers.

> Both datasets are **synthetic/simulated** and used for model development and
> demonstration only. This is a decision-support prototype — not a clinical
> device, and not an autonomous medical decision maker.

## Architecture

```
React (Vite/Framer Motion/Recharts)
        │  fetch + JWT
        ▼
FastAPI  ──►  PostgreSQL (resources, patients, forecasts, scenarios, optimization results)
   │
   ├── ML: patient deterioration risk (Random Forest + isotonic calibration)
   ├── ML: demand forecasting ICU / Emergency / General beds (Random Forest, 24/48/72h)
   ├── What-if simulation engine
   └── Google OR-Tools CP-SAT optimizer  →  explanation engine
```

The frontend design, layout, palette and navigation are unchanged; mock data
has been replaced with live API calls plus loading/error/retry states.

## Project structure

```
medicare-ai-frontend-v3/
├── medicare-ai-frontend/     # existing React app (src/main.jsx wired to the API)
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app + CORS
│   │   ├── api/              # auth, dashboard/resources, patients, forecast, scenarios
│   │   ├── database/         # SQLAlchemy models, session, seed
│   │   ├── services/         # risk / forecast / state services (inference)
│   │   ├── optimization/     # OR-Tools CP-SAT solver + explanation engine
│   │   ├── simulation/       # what-if scenario engine
│   │   ├── security.py       # JWT auth
│   │   └── config.py         # env-driven configuration
│   ├── training/             # train_patient_risk.py, train_demand_models.py
│   ├── data/raw/             # extracted datasets (Bed-prediction, hospital-deterioration)
│   ├── models/               # trained artifacts + metrics.json + metadata
│   ├── tests/test_api.py     # 13 backend tests
│   ├── requirements.txt
│   └── .env.example
└── docs/MODEL_REPORT.md      # full model report with real metrics
```

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # then edit: DATABASE_URL, ADMIN_PASSWORD, JWT_SECRET
```

### 2. Database (PostgreSQL)

Any PostgreSQL works — set `DATABASE_URL` in `backend/.env`, create the
database, then seed it:

```bash
createdb medicare
python -m app.database.seed   # loads resources, demand history, 10k patients
```

(On this machine a project-local cluster is used: `backend/data/pg/`, port 5433,
started with `pg_ctl -D backend/data/pg/data -o "-p 5433" -l server.log start`.)

### 3. Train models (reproduces all artifacts in `backend/models/`)

```bash
python training/train_patient_risk.py
python training/train_demand_models.py
```

### 4. Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```
Swagger: http://127.0.0.1:8000/docs

### 5. Frontend

```bash
cd medicare-ai-frontend
npm install
echo "VITE_API_URL=http://127.0.0.1:8000" > .env
npm run dev
```

Open http://localhost:5173 and sign in with `ADMIN_EMAIL` / `ADMIN_PASSWORD`
from `backend/.env` (default `admin@medicare.ai` / `medicare123` in the local
development `.env`; credentials are validated server-side and never baked
into frontend JavaScript).

## API (all under `/api`, JWT required except login/health)

| endpoint | purpose |
|---|---|
| POST `/auth/login` | JWT login |
| GET `/dashboard` | occupancy, ICU availability, ED demand, high-risk count, forecast |
| GET `/resources`, `/resources/beds` | live capacity + bed wall |
| GET `/patients`, `/patients/{id}/risk`, `/risk/summary` | model risk scores |
| GET `/forecast`, `/forecast/icu`, `/forecast/emergency`, `/forecast/beds` | 24/48/72h forecasts |
| POST `/simulation/run` | what-if scenario → pressure, shortages, optimized plan |
| POST `/optimization/run`, GET `/optimization/latest` | OR-Tools allocation + explanation |

## Tests

```bash
cd backend
python -m pytest tests/ -q     # 13 passed
```

## Security

`.env` (secrets, DB credentials, JWT secret) is gitignored; `.env.example` is
committed instead. No passwords exist in frontend source.

## Clinical safety

Patient-risk outputs are decision-support estimates from synthetic data for
demonstration purposes. The optimizer recommends resource allocation only — it
makes no clinical treatment decisions.
