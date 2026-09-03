"""Authentication middleware — protects page routes and API routes."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, JSONResponse
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED

from app.auth import decode_access_token
from app.config import settings

# Paths that don't require authentication
PUBLIC_PREFIXES = [
    "/login",
    "/static",
    "/health",
    "/api/v1/auth",
]

# Legacy API key — only valid for ingestion endpoints from edge devices
API_KEY_ENDPOINTS = [
    "/api/v1/episodes/start",
    "/api/v1/episodes",
    "/api/v1/sessions",
    "/api/v1/session/upload",
    "/api/v1/devices/heartbeat",
    "/api/v1/devices",
    "/api/v1/device/tasks",
    "/api/v1/worker",
]


def _is_api_key_path(path: str) -> bool:
    """Check if path is allowed with legacy API key."""
    for prefix in API_KEY_ENDPOINTS:
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            return True
    return False


def _is_public(path: str) -> bool:
    """Check if path is public (no auth required)."""
    for prefix in PUBLIC_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def _is_api_path(path: str) -> bool:
    return "/api/" in path


def _token_payload(request: Request) -> dict | None:
    token = request.cookies.get("auth_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        return None
    return decode_access_token(token)


def _has_valid_token(request: Request) -> bool:
    return _token_payload(request) is not None


def _has_valid_api_key(request: Request) -> bool:
    """Check legacy API key (for edge device ingestion)."""
    key = request.headers.get("X-API-Key", "")
    expected = settings.WORKER_API_KEY if request.url.path.startswith("/api/v1/worker") else settings.API_KEY
    expected = expected or settings.API_KEY
    if not expected or expected == "change-me":
        return True  # dev mode
    import secrets
    return secrets.compare_digest(key, expected)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if _is_public(path):
            return await call_next(request)

        # Expose non-sensitive token claims to the shared page layout so the
        # account area is correct on first paint. API dependencies still
        # revalidate the account in PostgreSQL before protected operations.
        token_payload = _token_payload(request)
        if token_payload:
            request.state.auth_user = token_payload
            return await call_next(request)

        # Fallback: legacy API key for ingestion endpoints only
        if _is_api_key_path(path) and _has_valid_api_key(request):
            return await call_next(request)

        # Unauthorized
        if _is_api_path(path):
            return JSONResponse(
                {"detail": "Not authenticated"},
                status_code=HTTP_401_UNAUTHORIZED,
            )

        # Page route — redirect to login
        return RedirectResponse(url="/login", status_code=302)
