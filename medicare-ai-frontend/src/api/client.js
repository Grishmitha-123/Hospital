// API client for the MediCare AI FastAPI backend.
// Backend URL comes from VITE_API_URL (never hardcoded).
const BASE = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

const TOKEN_KEY = 'medicare_token';

export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY);
}
export function setToken(t) {
  if (t) sessionStorage.setItem(TOKEN_KEY, t);
  else sessionStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

export async function request(path, { method = 'GET', body, signal } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (e) {
    throw new ApiError('Unable to reach the MediCare AI backend. Is it running?', 0);
  }
  if (res.status === 401) {
    setToken(null);
    throw new ApiError('Session expired — please sign in again.', 401);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* keep statusText */ }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

export const api = {
  login: (email, password) =>
    request('/api/auth/login', { method: 'POST', body: { email, password } }),
  health: () => request('/api/health'),
  dashboard: (signal) => request('/api/dashboard', { signal }),
  resources: (signal) => request('/api/resources', { signal }),
  beds: (signal) => request('/api/resources/beds', { signal }),
  patients: (limit = 25, level, signal) =>
    request(`/api/patients?limit=${limit}${level ? `&level=${level}` : ''}`, { signal }),
  patientRisk: (id, signal) => request(`/api/patients/${id}/risk`, { signal }),
  riskSummary: (signal) => request('/api/risk/summary', { signal }),
  forecast: (horizon = 72, signal) => request(`/api/forecast?horizon=${horizon}`, { signal }),
  simulationPresets: (signal) => request('/api/simulation/presets', { signal }),
  runSimulation: (payload) => request('/api/simulation/run', { method: 'POST', body: payload }),
  runOptimization: (payload) => request('/api/optimization/run', { method: 'POST', body: payload }),
  latestOptimization: (signal) => request('/api/optimization/latest', { signal }),
};

export default api;
