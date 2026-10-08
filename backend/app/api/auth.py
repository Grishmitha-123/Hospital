"""Auth endpoints: POST /api/auth/login (env-configured admin, JWT)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..security import authenticate, create_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    email: str = Field(..., examples=["admin@medicare.ai"])
    password: str


@router.post("/login")
def login(body: LoginIn):
    """Exchange credentials for a bearer token."""
    user = authenticate(body.email, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(user), "token_type": "bearer",
            "user": {"email": user["sub"], "role": user["role"]}}


@router.get("/whoami")
def whoami():
    """Public check used by the frontend to probe backend availability."""
    return {"status": "ok", "service": "medicare-ai"}
