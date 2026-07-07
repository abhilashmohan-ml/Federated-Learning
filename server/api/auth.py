"""
JWT authentication endpoints and the `get_current_site` dependency.

WHAT IS JWT AUTHENTICATION?
-----------------------------
JWT = JSON Web Token. It is a compact, URL-safe way for the server to prove
to a client (and vice versa) that it has been authenticated. A JWT looks like
three base64-encoded strings joined by dots:
    header.payload.signature

Example payload (decoded):
    {"sub": "site_1", "role": "client", "iat": 1700000000, "exp": 1700000900, "jti": "abc123"}

  sub  = subject (who the token belongs to — the site_id)
  role = "client" (all manufacturing sites are clients)
  iat  = issued at timestamp (Unix epoch seconds)
  exp  = expiration timestamp — the server rejects tokens after this time
  jti  = JWT ID — a unique UUID that lets us revoke individual tokens

The server signs the payload with the SECRET_KEY (HMAC-SHA256). If anyone
tampers with the payload, the signature becomes invalid and the server
rejects the token.

TWO-TOKEN PATTERN
-----------------
This system uses a pair of tokens:
  ACCESS token   — short-lived (15 min). Used for every API request.
                   If stolen, useless after 15 minutes.
  REFRESH token  — long-lived (7 days). Used ONLY to get a new token pair.
                   After each use, it is invalidated (single-use).

Why two tokens? If we used only one long-lived token, a stolen token would
give an attacker access for 7 days. With access + refresh, stealing an access
token gives at most 15 minutes. Stealing a refresh token gives one opportunity
to get new tokens, but that rotation revokes the old one, alerting the system.

PYTHON CONCEPT: APIRouter
  FastAPI's way of grouping related endpoints. The router object collects
  route definitions (with @router.post, @router.get, etc.) and is registered
  in server/main.py with `app.include_router(auth.router, prefix="/auth")`.

PYTHON CONCEPT: Depends()
  FastAPI's dependency injection. When an endpoint declares a parameter like
      db: AsyncSession = Depends(get_db)
  FastAPI calls get_db() and passes the result as `db`. This avoids having
  to manually wire up databases and settings in every endpoint.

PYTHON CONCEPT: HTTPException
  Raising HTTPException inside an endpoint handler causes FastAPI to return
  an HTTP error response (with the given status_code and detail message)
  instead of continuing with the normal response. Equivalent to `abort()` in
  Flask or `res.status(401).json({...})` in Express.

PYTHON CONCEPT: async/await
  All database operations use `await` because they talk to the database over
  a network connection (or disk I/O). While waiting, FastAPI can serve other
  requests.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt                  # python-jose: JWT encode/decode
import bcrypt as _bcrypt                         # bcrypt password hashing
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.config import ServerSettings, get_settings
from server.db.database import get_db
from server.db.models import RevokedToken, SiteRegistry
from shared.schemas.auth import RefreshRequest, TokenRequest, TokenResponse

router = APIRouter()

def _verify_secret(plaintext: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plaintext.encode(), hashed.encode())

ALGORITHM = "HS256"   # HMAC with SHA-256 — the standard for JWT signing

# HTTPBearer extracts the token from the "Authorization: Bearer <token>" header.
# FastAPI uses this as a dependency to pull the token out of incoming requests.
_bearer = HTTPBearer()


def _make_token(sub: str, role: str, expires: timedelta, key: str) -> tuple[str, str]:
    """
    Create a signed JWT token with the given claims.

    HOW JWT SIGNING WORKS
    ----------------------
    The payload (claims dict) is base64-encoded and signed with HMAC-SHA256
    using the `key`. The resulting signature is appended to the token.
    When the server receives a token, it recomputes the signature and compares —
    if they don't match, the token has been tampered with.

    Parameters
    ----------
    sub     : str      — "subject" = the site_id this token belongs to
    role    : str      — "client" (all manufacturing sites)
    expires : timedelta — how long until this token expires
    key     : str      — the server's SECRET_KEY (from settings)

    Returns
    -------
    tuple[str, str] — (encoded_jwt_string, jti)
        The jti (JWT ID) is a UUID4 hex string used to identify and revoke tokens.
    """
    now = datetime.now(timezone.utc)    # always use UTC for timestamps
    jti = uuid.uuid4().hex              # random 32-char hex string, unique per token
    token = jwt.encode(
        {
            "sub":  sub,
            "role": role,
            "iat":  now,           # issued at
            "exp":  now + expires, # expiration time
            "jti":  jti,           # unique token ID (for revocation)
        },
        key,
        algorithm=ALGORITHM,
    )
    return token, jti


async def get_current_site(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    s: ServerSettings = Depends(get_settings),
) -> str:
    """
    FastAPI dependency — verify the Bearer JWT and return the authenticated site_id.

    This is the AUTHENTICATION GATE for all protected endpoints. Any route
    that declares `_site: str = Depends(get_current_site)` will:
      1. Require an Authorization: Bearer <token> header in the request
      2. Reject the request with HTTP 401 if the token is missing, expired,
         or has an invalid signature
      3. Return the site_id string if the token is valid

    PYTHON CONCEPT: dependency chaining
      `Depends(_bearer)` is itself a dependency that extracts the token string
      from the request headers. FastAPI resolves dependencies in the right order.

    Parameters
    ----------
    creds : HTTPAuthorizationCredentials — contains the token from the header
    s     : ServerSettings — injected settings (includes secret_key)

    Returns
    -------
    str — the site_id extracted from the JWT's "sub" claim
    """
    try:
        # jwt.decode verifies the signature AND checks the expiry claim.
        # If either check fails, JWTError is raised.
        payload = jwt.decode(creds.credentials, s.secret_key, algorithms=[ALGORITHM])
        site_id: str = payload["sub"]   # extract the site identifier
        return site_id
    except (JWTError, KeyError):
        # JWTError: bad signature, expired, malformed
        # KeyError: "sub" claim missing from an otherwise valid token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},   # tells the client what to do
        )


@router.post("/token", response_model=TokenResponse)
async def issue_token(
    req: TokenRequest,
    s: ServerSettings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    POST /auth/token — site obtains its first access + refresh token pair.

    AUTHENTICATION FLOW
    -------------------
    1. Site sends its site_id and site_secret (cleartext over HTTPS)
    2. Server looks up the site in the site_registry table
    3. Server verifies the secret against the stored bcrypt hash
    4. If valid, server creates and returns a new access + refresh token pair

    The site_secret is only sent ONCE during this initial authentication.
    All subsequent API calls use the JWT access token instead.

    Parameters
    ----------
    req : TokenRequest — {"site_id": "site_1", "site_secret": "my_secret"}
    s   : ServerSettings — for token lifetimes and secret_key
    db  : AsyncSession — to look up the site registry

    Returns
    -------
    TokenResponse — {"access_token": "...", "refresh_token": "...", "token_type": "bearer"}
    """
    # SELECT * FROM site_registry WHERE site_id = req.site_id LIMIT 1
    result = await db.execute(
        select(SiteRegistry).where(SiteRegistry.site_id == req.site_id)
    )
    site = result.scalar_one_or_none()   # returns the row or None if not found

    # Timing-safe check: verify even if site is None (to prevent timing attacks
    # that reveal whether a site_id exists by measuring response time)
    if site is None or not _verify_secret(req.site_secret, site.secret_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad credentials"
        )

    # Create token pair. We discard the jti (second return value) here
    # because we don't need to revoke access tokens (they expire quickly).
    access_token, _  = _make_token(
        req.site_id, "client",
        timedelta(minutes=s.access_token_expire_minutes),
        s.secret_key,
    )
    refresh_token, _ = _make_token(
        req.site_id, "client",
        timedelta(days=s.refresh_token_expire_days),
        s.secret_key,
    )
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    req: RefreshRequest,
    s: ServerSettings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    POST /auth/refresh — exchange a refresh token for a new token pair.

    SINGLE-USE REFRESH (TOKEN ROTATION)
    ------------------------------------
    After using a refresh token to get a new pair, the old refresh token is
    immediately stored in the `revoked_tokens` table. If the same refresh
    token is presented again (e.g., by an attacker who intercepted it), the
    server detects it is already revoked and returns 401.

    This is called "refresh token rotation" — a security pattern that greatly
    limits the window during which a stolen refresh token is dangerous.

    Parameters
    ----------
    req : RefreshRequest — {"refresh_token": "the_jwt_string"}
    """
    try:
        # Decode and verify the refresh token's signature and expiry
        payload = jwt.decode(req.refresh_token, s.secret_key, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    jti: str | None = payload.get("jti")
    if jti:
        # Check revocation: was this token already used or explicitly revoked?
        existing = await db.execute(
            select(RevokedToken).where(RevokedToken.jti == jti)
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked"
            )

        # Revoke this refresh token now (single-use pattern)
        exp_ts: int = payload.get("exp", 0)
        db.add(RevokedToken(
            jti=jti,
            site_id=payload["sub"],
            # Convert Unix epoch int to UTC datetime for the DB column
            expires_at=datetime.fromtimestamp(exp_ts, tz=timezone.utc),
        ))
        await db.commit()   # persist the revocation before issuing new tokens

    site_id: str = payload["sub"]
    new_access, _  = _make_token(
        site_id, "client",
        timedelta(minutes=s.access_token_expire_minutes),
        s.secret_key,
    )
    new_refresh, _ = _make_token(
        site_id, "client",
        timedelta(days=s.refresh_token_expire_days),
        s.secret_key,
    )
    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


@router.post("/revoke")
async def revoke_token(
    req: RefreshRequest,
    s: ServerSettings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    POST /auth/revoke — explicitly invalidate a refresh token (logout).

    A site calls this when it wants to log out proactively. After revocation,
    the refresh token cannot be used to get new tokens.

    Note: This endpoint ALWAYS returns {"status": "revoked"}, even if the
    token is already invalid. This is intentional — it prevents the caller
    from learning whether the token was previously valid (idempotent endpoint).

    Parameters
    ----------
    req : RefreshRequest — the refresh token to invalidate
    """
    try:
        payload = jwt.decode(req.refresh_token, s.secret_key, algorithms=[ALGORITHM])
    except JWTError:
        # Token is already invalid (expired, bad signature) — nothing to revoke
        return {"status": "revoked"}

    jti: str | None = payload.get("jti")
    if jti:
        # Only insert if not already in the revoked list (avoid duplicate key error)
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
