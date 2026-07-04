"""
JWT authentication endpoints and get_current_site dependency.

POST /auth/token    site obtains access + refresh tokens
POST /auth/refresh  rotate tokens
POST /auth/revoke   invalidate refresh token
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.config import ServerSettings, get_settings
from server.db.database import get_db
from server.db.models import RevokedToken, SiteRegistry
from shared.schemas.auth import RefreshRequest, TokenRequest, TokenResponse

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"
_bearer = HTTPBearer()


def _make_token(sub: str, role: str, expires: timedelta, key: str) -> tuple[str, str]:
    """Return (encoded_jwt, jti)."""
    now = datetime.now(timezone.utc)
    jti = uuid.uuid4().hex
    token = jwt.encode(
        {"sub": sub, "role": role, "iat": now, "exp": now + expires, "jti": jti},
        key,
        algorithm=ALGORITHM,
    )
    return token, jti


async def get_current_site(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    s: ServerSettings = Depends(get_settings),
) -> str:
    """FastAPI dependency — verify Bearer JWT, return site_id."""
    try:
        payload = jwt.decode(creds.credentials, s.secret_key, algorithms=[ALGORITHM])
        site_id: str = payload["sub"]
        return site_id
    except (JWTError, KeyError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/token", response_model=TokenResponse)
async def issue_token(
    req: TokenRequest,
    s: ServerSettings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(
        select(SiteRegistry).where(SiteRegistry.site_id == req.site_id)
    )
    site = result.scalar_one_or_none()
    if site is None or not pwd_context.verify(req.site_secret, site.secret_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad credentials"
        )
    access_token, _ = _make_token(
        req.site_id, "client", timedelta(minutes=s.access_token_expire_minutes), s.secret_key
    )
    refresh_token, _ = _make_token(
        req.site_id, "client", timedelta(days=s.refresh_token_expire_days), s.secret_key
    )
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    req: RefreshRequest,
    s: ServerSettings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    try:
        payload = jwt.decode(req.refresh_token, s.secret_key, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    jti: str | None = payload.get("jti")
    if jti:
        existing = await db.execute(
            select(RevokedToken).where(RevokedToken.jti == jti)
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked"
            )
        # Revoke consumed refresh token before issuing new pair
        exp_ts: int = payload.get("exp", 0)
        db.add(RevokedToken(
            jti=jti,
            site_id=payload["sub"],
            expires_at=datetime.fromtimestamp(exp_ts, tz=timezone.utc),
        ))
        await db.commit()

    site_id: str = payload["sub"]
    new_access, _ = _make_token(
        site_id, "client", timedelta(minutes=s.access_token_expire_minutes), s.secret_key
    )
    new_refresh, _ = _make_token(
        site_id, "client", timedelta(days=s.refresh_token_expire_days), s.secret_key
    )
    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


@router.post("/revoke")
async def revoke_token(
    req: RefreshRequest,
    s: ServerSettings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        payload = jwt.decode(req.refresh_token, s.secret_key, algorithms=[ALGORITHM])
    except JWTError:
        return {"status": "revoked"}  # already unusable
    jti: str | None = payload.get("jti")
    if jti:
        existing = await db.execute(
            select(RevokedToken).where(RevokedToken.jti == jti)
        )
        if existing.scalar_one_or_none() is None:
            exp_ts: int = payload.get("exp", 0)
            db.add(RevokedToken(
                jti=jti,
                site_id=payload["sub"],
                expires_at=datetime.fromtimestamp(exp_ts, tz=timezone.utc),
            ))
            await db.commit()
    return {"status": "revoked"}
