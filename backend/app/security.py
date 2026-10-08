"""JWT authentication (admin-only, hackathon scope).

Credentials come from environment variables (see .env.example); nothing is
hardcoded in the frontend or in source.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import ADMIN_EMAIL, ADMIN_PASSWORD, JWT_ALGORITHM, JWT_EXPIRE_MINUTES

_bearer = HTTPBearer(auto_error=False)


def _secret() -> str:
    import os
    s = os.getenv("JWT_SECRET", "")
    if s:
        return s
    # Ephemeral development secret: process-local, never written to source.
    return secrets.token_hex(32)


def authenticate(email: str, password: str) -> dict | None:
    """Constant-time credential check against env-configured admin account."""
    if not ADMIN_PASSWORD:
        return None
    ok_email = secrets.compare_digest(email.strip().lower(),
                                      ADMIN_EMAIL.strip().lower())
    ok_pw = secrets.compare_digest(password, ADMIN_PASSWORD)
    if ok_email and ok_pw:
        return {"sub": email.strip().lower(), "role": "admin"}
    return None


def create_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user["sub"],
        "role": user["role"],
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGORITHM)


def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            detail="Not authenticated")
    try:
        return jwt.decode(creds.credentials, _secret(),
                          algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired token")
