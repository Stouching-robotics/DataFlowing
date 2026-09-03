"""Authentication API — EGOData PostgreSQL accounts with legacy fallback."""

import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from app.auth import (
    hash_password, verify_password, create_access_token, get_current_user,
    authenticate_db, _is_expired, ROLE_PERMISSIONS,
)
from app.localstore import STATE_ROOT

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

USERS_FILE = STATE_ROOT / "users.json"

# Bootstrap account for a new legacy file store. Existing users.json is kept.
_BOOTSTRAP_USERNAME = (os.environ.get("EGODATA_BOOTSTRAP_USERNAME", "demo-admin").strip()
                       or "demo-admin")
_BOOTSTRAP_PASSWORD = os.environ.get("EGODATA_BOOTSTRAP_PASSWORD", "change-me")
_BOOTSTRAP_EMAIL = (os.environ.get("EGODATA_BOOTSTRAP_EMAIL", "").strip()
                    or f"{_BOOTSTRAP_USERNAME.lower()}@egodata.local")

_DEFAULT_USERS = [
    {"id": "local-admin", "username": _BOOTSTRAP_USERNAME,
     "password_hash": None, "role": "admin", "email": _BOOTSTRAP_EMAIL,
     "status": "active"},
]


def _load_users() -> list[dict]:
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        users = []
        for u in _DEFAULT_USERS:
            u = dict(u)
            if not u["password_hash"]:
                u["password_hash"] = hash_password(_BOOTSTRAP_PASSWORD)
            users.append(u)
        try:
            USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
            USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass
        return users


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)
    remember_me: bool = Field(default=False)


@router.post("/login")
async def login(body: LoginRequest, response: Response):
    """Authenticate against EGOData users, with legacy JSON fallback."""
    db_user, db_available = await authenticate_db(body.username, body.password)
    if db_available:
        if db_user is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user = {
            "id": str(db_user.id), "username": db_user.username,
            "role": db_user.role, "email": db_user.email,
            "expires_at": db_user.expires_at.isoformat() if db_user.expires_at else None,
        }
    else:
        users = _load_users()
        user = next(
            (u for u in users
             if u.get("username") == body.username
             and u.get("status") == "active"
             and not _is_expired(u.get("expires_at"))),
            None,
        )
        if not user or not verify_password(body.password, user.get("password_hash", "")):
            raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(user["id"]), user["username"], user["role"])

    cookie_kwargs = {
        "key": "auth_token",
        "value": token,
        "httponly": True,
        "samesite": "lax",
        "secure": False,
        "path": "/",
    }
    if body.remember_me:
        cookie_kwargs["max_age"] = 30 * 24 * 3600
    response.set_cookie(**cookie_kwargs)

    return {
        "message": "Login successful",
        "user": {
            "id": str(user["id"]),
            "username": user["username"],
            "role": user["role"],
            "email": user.get("email"),
        },
    }

@router.post("/logout")
async def logout(response: Response):
    """Clear auth cookie."""
    response.delete_cookie("auth_token", path="/")
    return {"message": "Logged out"}


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return current authenticated user info."""
    return {
        "id": current_user["sub"],
        "username": current_user["username"],
        "role": current_user["role"],
        "email": current_user.get("email"),
        "expires_at": current_user.get("expires_at"),
        "permissions": sorted(ROLE_PERMISSIONS.get(current_user["role"], set())),
    }
