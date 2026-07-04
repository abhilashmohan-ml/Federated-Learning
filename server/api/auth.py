"""
JWT authentication endpoints.

POST /auth/token    site obtains access + refresh tokens
POST /auth/refresh  rotate tokens
POST /auth/revoke   invalidate refresh token
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from server.config import ServerSettings, get_settings
from shared.schemas.auth import TokenRequest, TokenResponse, RefreshRequest

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"

# ── In-memory site registry (replace with DB in production) ───────────────────
_SITE_SECRETS: dict[str, str] = {f"site_{i}": f"secret_site_{i}" for i in range(1, 6)}
_REVOKED: set[str] = set()


def _make_token(sub: str, role: str, expires: timedelta, key: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": sub, "role": role, "iat": now, "exp": now + expires},
        key, algorithm=ALGORITHM,
    )


@router.post("/token", response_model=TokenResponse)
async def issue_token(
    req: TokenRequest,
    s: ServerSettings = Depends(get_settings),
) -> TokenResponse:
    expected = _SITE_SECRETS.get(req.site_id)
    if not expected or expected != req.site_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad credentials")
    return TokenResponse(
        access_token  = _make_token(req.site_id, "client", timedelta(minutes=s.access_token_expire_minutes), s.secret_key),
        refresh_token = _make_token(req.site_id, "client", timedelta(days=s.refresh_token_expire_days), s.secret_key),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    req: RefreshRequest,
    s: ServerSettings = Depends(get_settings),
) -> TokenResponse:
    if req.refresh_token in _REVOKED:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")
    try:
        payload = jwt.decode(req.refresh_token, s.secret_key, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    _REVOKED.add(req.refresh_token)
    site_id: str = payload["sub"]
    return TokenResponse(
        access_token  = _make_token(site_id, "client", timedelta(minutes=s.access_token_expire_minutes), s.secret_key),
        refresh_token = _make_token(site_id, "client", timedelta(days=s.refresh_token_expire_days), s.secret_key),
    )


@router.post("/revoke")
async def revoke_token(req: RefreshRequest) -> dict:
    _REVOKED.add(req.refresh_token)
    return {"status": "revoked"}
