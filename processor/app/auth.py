"""Authentication utilities — bcrypt password hashing + JWT tokens."""

import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request

# ── JWT secret ────────────────────────────────────────────
_JWT_SECRET = os.environ.get(
    "JWT_SECRET",
    "egodata-v1-jwt-production-secret-key-2026"
)
_JWT_ALGORITHM = "HS256"
# Keep the access-token lifetime configurable while preserving the current
# login API.  Thirty days prevents the browser from losing its authenticated
# state during normal workstation restarts; refresh-token rotation can be
# introduced later for a shorter-lived access token model.
try:
    _JWT_EXPIRE_DAYS = max(1, int(os.environ.get("JWT_EXPIRE_DAYS", "30")))
except ValueError:
    _JWT_EXPIRE_DAYS = 30
_TOKEN_EXPIRE_HOURS = _JWT_EXPIRE_DAYS * 24

ROLE_PERMISSIONS = {
    "admin": {
        "users.manage", "projects.manage", "workflows.manage",
        "processing.run", "review.read", "review.write", "system.manage",
    },
    "engineer": {
        "projects.manage", "workflows.manage", "processing.run",
        "review.read", "review.write",
    },
    "reviewer": {"review.read", "review.write"},
}


# ── Password hashing (HMAC-SHA256 with per-user salt) ─────

def _derive_key(password: str, salt: str) -> bytes:
    """Derive a key from password and salt using PBKDF2."""
    import hashlib
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)


def hash_password(plain: str) -> str:
    """Hash a password with random salt. Returns 'salt$hex_digest'."""
    salt = os.urandom(16).hex()
    key = _derive_key(plain, salt)
    return f"{salt}${key.hex()}"


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    try:
        salt, digest = hashed.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(_derive_key(plain, salt).hex(), digest)


# ── JWT ───────────────────────────────────────────────────

def _b64url_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    import base64
    # Add padding
    rem = len(s) % 4
    if rem:
        s += "=" * (4 - rem)
    return base64.urlsafe_b64decode(s)


def create_access_token(user_id: str, username: str, role: str) -> str:
    """Create a signed JWT access token (no external lib needed)."""
    header = _b64url_encode(json.dumps({"alg": _JWT_ALGORITHM, "typ": "JWT"}).encode())
    now = datetime.now(timezone.utc)
    payload = _b64url_encode(json.dumps({
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=_TOKEN_EXPIRE_HOURS)).timestamp()),
    }).encode())
    msg = f"{header}.{payload}".encode()
    sig = _b64url_encode(hmac.digest(_JWT_SECRET.encode(), msg, "sha256"))
    return f"{header}.{payload}.{sig}"


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT. Returns payload dict or None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        msg = f"{header_b64}.{payload_b64}".encode()
        expected_sig = _b64url_encode(hmac.digest(_JWT_SECRET.encode(), msg, "sha256"))
        if not hmac.compare_digest(sig_b64, expected_sig):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        # Check expiration
        exp = payload.get("exp", 0)
        if datetime.now(timezone.utc).timestamp() > exp:
            return None
        return payload
    except Exception:
        return None


# ── FastAPI dependency ─────────────────────────────────────

async def get_current_user(request: Request) -> dict:
    """Extract user from cookie or Authorization header. Returns payload dict."""
    token = None

    # Check cookie first
    token = request.cookies.get("auth_token")

    # Fallback to Authorization header
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # EGOData PostgreSQL is authoritative for managed accounts.  The legacy
    # users.json path remains a fallback so an old installation can still
    # start and accept its existing login until the DB is initialized.
    db_user, db_available = await _find_db_user(user_id=str(payload["sub"]))
    if db_available:
        if db_user is None or db_user.status != "active":
            raise HTTPException(status_code=401, detail="User disabled or deleted")
        if _is_expired(db_user.expires_at):
            raise HTTPException(status_code=401, detail="Account expired")
        payload.update(_user_public_dict(db_user))
        return payload

    from app.localstore import STATE_ROOT
    users_file = STATE_ROOT / "users.json"
    try:
        users = json.loads(users_file.read_text(encoding="utf-8"))
    except Exception:
        users = []
    user = next(
        (u for u in users if str(u.get("id")) == str(payload["sub"])
         and u.get("status") == "active"),
        None,
    )
    if not user:
        raise HTTPException(status_code=401, detail="User disabled or deleted")
    if _is_expired(user.get("expires_at")):
        raise HTTPException(status_code=401, detail="Account expired")
    payload.update({
        "username": user.get("username", payload.get("username")),
        "role": user.get("role", payload.get("role")),
        "email": user.get("email"),
        "id": str(user.get("id", payload["sub"])),
    })
    return payload


def _is_expired(value) -> bool:
    if not value:
        return False
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt <= datetime.now(timezone.utc)


def _user_public_dict(user) -> dict:
    return {
        "id": str(user.id),
        "username": user.username,
        "role": user.role,
        "email": user.email,
        "status": user.status,
        "expires_at": user.expires_at.isoformat() if user.expires_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }


async def _find_db_user(*, user_id: str | None = None, username: str | None = None):
    """Return (user, db_available); distinguish empty DB from DB failure."""
    try:
        from sqlalchemy import select
        from app.database import async_session
        from app.models import User
        async with async_session() as db:
            query = select(User)
            if user_id is not None:
                query = query.where(User.id == user_id)
            else:
                query = query.where(User.username == username)
            return (await db.execute(query)).scalar_one_or_none(), True
    except Exception as exc:
        # Keep legacy auth available during a temporary DB/tunnel outage.
        print(f"[Auth] Database lookup unavailable: {exc}")
        return None, False


async def authenticate_db(username: str, password: str):
    """Return (user, db_available) for login, updating last_login_at."""
    user, available = await _find_db_user(username=username)
    if not available or user is None:
        return (None, available)
    if user.status != "active" or _is_expired(user.expires_at):
        return (None, True)
    if not verify_password(password, user.password_hash):
        return (None, True)
    try:
        from app.database import async_session
        async with async_session() as db:
            managed = await db.get(type(user), user.id)
            if managed is not None:
                managed.last_login_at = datetime.now(timezone.utc)
                await db.commit()
                user = managed
    except Exception as exc:
        print(f"[Auth] Could not update last_login_at: {exc}")
    return (user, True)


def require_roles(*roles: str):
    """Build a FastAPI dependency for role-protected routes."""
    allowed = set(roles)

    async def dependency(current_user: dict = Depends(get_current_user)):
        if current_user.get("role") not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user

    return dependency
