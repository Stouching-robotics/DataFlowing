"""Administrator account management for the EGOData platform."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.auth import _is_expired, get_current_user, hash_password, require_roles
from app.database import async_session
from app.models import User

router = APIRouter(prefix="/api/v1/users", tags=["users"])

Role = Literal["admin", "reviewer", "engineer"]
UserStatus = Literal["active", "disabled"]


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    email: str | None = Field(default=None, max_length=128)
    role: Role = "engineer"
    expires_at: datetime | None = None


class UserUpdate(BaseModel):
    email: str | None = Field(default=None, max_length=128)
    role: Role | None = None
    status: UserStatus | None = None
    password: str | None = Field(default=None, min_length=6, max_length=128)
    expires_at: datetime | None = None


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _status(user: User) -> str:
    return "expired" if user.status == "active" and _is_expired(user.expires_at) else user.status


def _out(user: User) -> dict:
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "status": _status(user),
        "account_status": user.status,
        "expires_at": user.expires_at.isoformat() if user.expires_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


async def _get_user(db, user_id: str) -> User:
    try:
        user = await db.get(User, user_id)
    except Exception:
        user = None
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def _admin_count(db) -> int:
    return int((await db.execute(
        select(func.count(User.id)).where(User.role == "admin", User.status == "active")
    )).scalar() or 0)


@router.get("")
async def list_users(
    role: Role | None = Query(default=None),
    status: str | None = Query(default=None),
    _: dict = Depends(require_roles("admin")),
):
    async with async_session() as db:
        query = select(User).order_by(User.created_at.asc(), User.username.asc())
        if role:
            query = query.where(User.role == role)
        if status in {"active", "disabled"}:
            query = query.where(User.status == status)
        users = (await db.execute(query)).scalars().all()
        return {"users": [_out(user) for user in users], "total": len(users)}


@router.post("", status_code=201)
async def create_user(body: UserCreate, _: dict = Depends(require_roles("admin"))):
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="Username cannot be empty")
    async with async_session() as db:
        exists = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
        if exists:
            raise HTTPException(status_code=409, detail="Username already exists")
        user = User(
            username=username,
            password_hash=hash_password(body.password),
            email=str(body.email) if body.email else None,
            role=body.role,
            status="active",
            expires_at=_utc(body.expires_at),
        )
        db.add(user)
        try:
            await db.commit()
            await db.refresh(user)
        except IntegrityError:
            await db.rollback()
            raise HTTPException(status_code=409, detail="Username already exists")
        return _out(user)


@router.patch("/{user_id}")
async def update_user(
    user_id: str,
    body: UserUpdate,
    current_user: dict = Depends(require_roles("admin")),
):
    async with async_session() as db:
        user = await _get_user(db, user_id)
        if body.status == "disabled" and str(user.id) == str(current_user.get("sub")):
            raise HTTPException(status_code=400, detail="You cannot disable your own account")
        if body.role == "admin" and user.role != "admin":
            # No special action; kept explicit for readability of the audit path.
            pass
        if user.role == "admin" and body.role and body.role != "admin" and user.status == "active":
            if await _admin_count(db) <= 1:
                raise HTTPException(status_code=400, detail="At least one active admin is required")
        if user.role == "admin" and body.status == "disabled" and user.status == "active":
            if await _admin_count(db) <= 1:
                raise HTTPException(status_code=400, detail="At least one active admin is required")

        if "email" in body.model_fields_set:
            user.email = str(body.email) if body.email else None
        if body.role is not None:
            user.role = body.role
        if body.status is not None:
            user.status = body.status
        if body.password:
            user.password_hash = hash_password(body.password)
        if "expires_at" in body.model_fields_set:
            user.expires_at = _utc(body.expires_at)
        await db.commit()
        await db.refresh(user)
        return _out(user)


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    current_user: dict = Depends(require_roles("admin")),
):
    """Permanently remove an account; UI primarily uses disable instead."""
    async with async_session() as db:
        user = await _get_user(db, user_id)
        if str(user.id) == str(current_user.get("sub")):
            raise HTTPException(status_code=400, detail="You cannot delete your own account")
        if user.role == "admin" and user.status == "active" and await _admin_count(db) <= 1:
            raise HTTPException(status_code=400, detail="At least one active admin is required")
        await db.delete(user)
        await db.commit()
        return {"message": "User deleted"}
