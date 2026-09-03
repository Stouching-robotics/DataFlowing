"""API Key authentication dependency."""

import secrets
from fastapi import HTTPException, Header
from app.config import settings


async def verify_api_key(x_api_key: str = Header(default="")) -> str:
    """Verify the X-API-Key header against the configured key.

    Skips verification if API_KEY is empty (dev mode).
    """
    if not settings.API_KEY or settings.API_KEY == "change-me":
        return "dev"
    if not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return "ok"


async def verify_worker_api_key(x_api_key: str = Header(default="")) -> str:
    """Verify the key used by local post-processing workers."""
    expected = settings.WORKER_API_KEY or settings.API_KEY
    if not expected or expected == "change-me":
        return "dev"
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid worker API key")
    return "ok"
