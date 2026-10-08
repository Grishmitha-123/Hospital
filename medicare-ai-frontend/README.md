# MediCare AI — Enhanced Frontend

A polished React + Vite prototype for the MediCare AI hospital resource intelligence platform.

## Included
- Admin login gate (prototype/demo authentication)
- Light / dark mode
- Animated hospital imagery
- Command Center dashboard
- Bed-by-bed occupancy wall with occupied, reserved, available and cleaning states
- Patient risk intelligence
- 24/48/72-hour demand forecasting
- Interactive what-if simulation
- Constraint-aware resource optimization
- Predict → Simulate → Optimize → Explain visual flow
- Recharts visualizations
- Framer Motion animations
- Lucide React icons

## Run locally
```bash
npm install
npm run dev
```

Then open the local Vite URL.

### Login
Authentication is handled by the FastAPI backend (JWT). Use the admin
credentials configured in `backend/.env` (`ADMIN_EMAIL` / `ADMIN_PASSWORD`);
the local development defaults are `admin@medicare.ai` / `medicare123`.

## Backend integration
All data on every page comes from the FastAPI backend (`VITE_API_URL` in
`.env`, see `src/api/client.js`): dashboard metrics, bed wall, patient risk
scores, 24/48/72h forecasts, what-if simulations and OR-Tools allocations.
Loading, error and retry states are built in; failed requests never fall back
to fake numbers.
