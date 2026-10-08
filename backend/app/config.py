"""Application configuration from environment variables (.env supported)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://medicare:medicare_local_dev@127.0.0.1:5433/medicare",
)

# Auth (never hardcode secrets in source; see .env.example)
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@medicare.ai")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# CORS for the Vite dev server
CORS_ORIGINS = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]

# Auto-refresh interval hint for the frontend (ms)
REFRESH_INTERVAL_MS = int(os.getenv("REFRESH_INTERVAL_MS", "60000"))
