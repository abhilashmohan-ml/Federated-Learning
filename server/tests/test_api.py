"""Unit tests for server/api — 100% coverage."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import bcrypt as _bcrypt
import httpx
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from server.api.auth import ALGORITHM, _make_token, get_current_site
from server.config import get_settings
from server.core.round_manager import get_round_manager
from server.db.database import Base, get_db
from server.db.models import RevokedToken, SiteRegistry
from server.main import app
from shared.schemas.federation import FederationRound, RoundStatus

# ── In-memory DB helpers ────────────────────────────────────────────────────────

TEST_DB = "sqlite+aiosqlite:///:memory:"


async def _make_session():
    """Create a fresh in-memory SQLite engine with all tables; return a session factory."""
    eng = create_async_engine(TEST_DB)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(eng, expire_on_commit=False)


def _db_override(session_local):
    """Return an async generator suitable for overriding get_db."""
    async def _get_db():
        async with session_local() as s:
            yield s

    return _get_db


# ── Auth header helper ──────────────────────────────────────────────────────────

def _auth_headers(site_id: str = "site_1") -> dict:
    """Return Authorization headers with a valid short-lived JWT for *site_id*."""
    s = get_settings()
    token, _ = _make_token(site_id, "client", timedelta(minutes=15), s.secret_key)
    return {"Authorization": f"Bearer {token}"}


# ── Federation mock helper ──────────────────────────────────────────────────────

def _fed_round(round_id: int = 1) -> FederationRound:
    return FederationRound(
        round_id=round_id,
        status=RoundStatus.COLLECTING,
        started_at=datetime.now(timezone.utc),
    )


def _mock_rm() -> MagicMock:
    """Return a MagicMock pre-configured to look like a RoundManager."""
    rm = MagicMock()
    rm.start_new_round = AsyncMock(return_value=_fed_round(1))
    rm.receive_update = AsyncMock(return_value=None)
    rm.get_round = AsyncMock(return_value=_fed_round(1))
    rm.get_site_statuses = AsyncMock(return_value={"site_1": "idle"})
    rm.current_global_weights = {}
    return rm


# ══ GET /health/ and GET /health/metrics ═════════════════════════════════════════

class TestHealth:
    def test_liveness_returns_ok(self) -> None:
        async def _run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.get("/health/")

        r = asyncio.run(_run())
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_metrics_returns_zero_counters(self) -> None:
        async def _run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.get("/health/metrics")

        r = asyncio.run(_run())
        assert r.status_code == 200
        assert r.json() == {"rounds_completed": 0, "sites_connected": 0}


# ══ GET /models/global-model ═════════════════════════════════════════════════════

class TestModels:
    def test_no_weights_returns_message(self) -> None:
        rm = _mock_rm()
        rm.current_global_weights = {}
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.get(
                        "/models/global-model", headers=_auth_headers()
                    )

            r = asyncio.run(_run())
            assert r.status_code == 200
            assert r.json() == {"message": "No global model available yet"}
        finally:
            app.dependency_overrides.clear()

    def test_weights_present_returns_weights(self) -> None:
        weights = {"hermia_params": [1.0, 2.0, 3.0]}
        rm = _mock_rm()
        rm.current_global_weights = weights
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.get(
                        "/models/global-model", headers=_auth_headers()
                    )

            r = asyncio.run(_run())
            assert r.status_code == 200
            assert r.json() == weights
        finally:
            app.dependency_overrides.clear()


# ══ Federation endpoints ══════════════════════════════════════════════════════════

class TestFederation:
    def test_start_round_success(self) -> None:
        rm = _mock_rm()
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/federation/round/start", headers=_auth_headers()
                    )

            r = asyncio.run(_run())
            assert r.status_code == 200
            data = r.json()
            assert data["round_id"] == 1
            assert data["status"] == "collecting"
            rm.start_new_round.assert_awaited_once()
        finally:
            app.dependency_overrides.clear()

    def test_receive_update_success(self) -> None:
        rm = _mock_rm()
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/federation/update",
                        headers=_auth_headers("site_1"),
                        json={
                            "site_id": "site_1",
                            "round_id": 1,
                            "n_samples": 100,
                            "delta_W": {"layer_a": [0.1, 0.2]},
                        },
                    )

            r = asyncio.run(_run())
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "accepted"
            assert body["site_id"] == "site_1"
            assert body["round_id"] == 1
        finally:
            app.dependency_overrides.clear()

    def test_receive_update_site_id_mismatch_returns_403(self) -> None:
        rm = _mock_rm()
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    # JWT identifies site_1, body claims site_2 → mismatch
                    return await client.post(
                        "/federation/update",
                        headers=_auth_headers("site_1"),
                        json={
                            "site_id": "site_2",
                            "round_id": 1,
                            "n_samples": 50,
                            "delta_W": {"layer_a": [0.5]},
                        },
                    )

            r = asyncio.run(_run())
            assert r.status_code == 403
            assert "mismatch" in r.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_get_round_found(self) -> None:
        rm = _mock_rm()
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.get(
                        "/federation/round/1", headers=_auth_headers()
                    )

            r = asyncio.run(_run())
            assert r.status_code == 200
            assert r.json()["round_id"] == 1
        finally:
            app.dependency_overrides.clear()

    def test_get_round_not_found_returns_404(self) -> None:
        rm = _mock_rm()
        rm.get_round = AsyncMock(return_value=None)
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.get(
                        "/federation/round/999", headers=_auth_headers()
                    )

            r = asyncio.run(_run())
            assert r.status_code == 404
        finally:
            app.dependency_overrides.clear()

    def test_list_sites_returns_dict(self) -> None:
        rm = _mock_rm()
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.get(
                        "/federation/sites", headers=_auth_headers()
                    )

            r = asyncio.run(_run())
            assert r.status_code == 200
            body = r.json()
            assert "sites" in body
            assert body["sites"] == {"site_1": "idle"}
        finally:
            app.dependency_overrides.clear()


# ══ Auth — _make_token (unit) ════════════════════════════════════════════════════

class TestMakeToken:
    def test_returns_pair_of_non_empty_strings(self) -> None:
        s = get_settings()
        token, jti = _make_token("site_1", "client", timedelta(minutes=15), s.secret_key)
        assert isinstance(token, str) and token
        assert isinstance(jti, str) and jti

    def test_jti_is_32_char_hex(self) -> None:
        s = get_settings()
        _, jti = _make_token("site_1", "client", timedelta(minutes=5), s.secret_key)
        assert len(jti) == 32
        # Should parse as hex without error
        int(jti, 16)

    def test_token_encodes_expected_claims(self) -> None:
        s = get_settings()
        token, jti = _make_token("site_42", "client", timedelta(minutes=5), s.secret_key)
        payload = jwt.decode(token, s.secret_key, algorithms=[ALGORITHM])
        assert payload["sub"] == "site_42"
        assert payload["role"] == "client"
        assert payload["jti"] == jti


# ══ Auth — get_current_site dependency (unit) ════════════════════════════════════

class TestGetCurrentSite:
    def test_valid_token_returns_site_id(self) -> None:
        async def _run():
            s = get_settings()
            token, _ = _make_token("site_3", "client", timedelta(minutes=15), s.secret_key)
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            return await get_current_site(creds=creds, s=s)

        assert asyncio.run(_run()) == "site_3"

    def test_invalid_token_raises_401(self) -> None:
        async def _run():
            s = get_settings()
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="not.a.real.jwt"
            )
            return await get_current_site(creds=creds, s=s)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 401

    def test_missing_sub_claim_raises_401(self) -> None:
        """Valid signature but absent 'sub' field raises KeyError → caught as 401."""
        async def _run():
            s = get_settings()
            now = datetime.now(timezone.utc)
            # Craft a valid-signature JWT that deliberately omits 'sub'
            token = jwt.encode(
                {"role": "client", "exp": now + timedelta(minutes=5)},
                s.secret_key,
                algorithm=ALGORITHM,
            )
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            return await get_current_site(creds=creds, s=s)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 401


# ══ Auth — POST /auth/token ═══════════════════════════════════════════════════════

class TestIssueToken:
    def test_success(self) -> None:
        async def _run():
            session_local = await _make_session()
            hashed = _bcrypt.hashpw(b"s3cr3t", _bcrypt.gensalt()).decode()
            async with session_local() as s:
                s.add(SiteRegistry(site_id="site_1", secret_hash=hashed))
                await s.commit()
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/token",
                        json={"site_id": "site_1", "site_secret": "s3cr3t"},
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    def test_site_not_found_returns_401(self) -> None:
        async def _run():
            session_local = await _make_session()
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/token",
                        json={"site_id": "ghost", "site_secret": "anything"},
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 401
        assert "credentials" in r.json()["detail"].lower()

    def test_wrong_password_returns_401(self) -> None:
        async def _run():
            session_local = await _make_session()
            hashed = _bcrypt.hashpw(b"correct_pw", _bcrypt.gensalt()).decode()
            async with session_local() as s:
                s.add(SiteRegistry(site_id="site_1", secret_hash=hashed))
                await s.commit()
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/token",
                        json={"site_id": "site_1", "site_secret": "wrong_pw"},
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 401


# ══ Auth — POST /auth/refresh ═════════════════════════════════════════════════════

class TestRefreshToken:
    def test_success_issues_new_token_pair(self) -> None:
        async def _run():
            session_local = await _make_session()
            s = get_settings()
            refresh_tok, _ = _make_token(
                "site_1", "client", timedelta(days=7), s.secret_key
            )
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/refresh", json={"refresh_token": refresh_tok}
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert "refresh_token" in body

    def test_invalid_jwt_returns_401(self) -> None:
        async def _run():
            session_local = await _make_session()
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/refresh", json={"refresh_token": "bad.jwt.here"}
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 401

    def test_already_revoked_returns_401(self) -> None:
        async def _run():
            session_local = await _make_session()
            s = get_settings()
            refresh_tok, jti = _make_token(
                "site_1", "client", timedelta(days=7), s.secret_key
            )
            # Pre-populate the revocation table so the token is already spent
            async with session_local() as sess:
                sess.add(RevokedToken(
                    jti=jti,
                    site_id="site_1",
                    expires_at=datetime.now(timezone.utc) + timedelta(days=7),
                ))
                await sess.commit()
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/refresh", json={"refresh_token": refresh_tok}
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 401
        assert "revoked" in r.json()["detail"].lower()

    def test_token_without_jti_skips_revocation_check(self) -> None:
        """Covers the `if jti:` False branch — token with no jti is accepted."""
        async def _run():
            session_local = await _make_session()
            s = get_settings()
            now = datetime.now(timezone.utc)
            token_no_jti = jwt.encode(
                {
                    "sub": "site_1",
                    "role": "client",
                    "iat": now,
                    "exp": now + timedelta(days=7),
                    # Note: deliberately no "jti" field
                },
                s.secret_key,
                algorithm=ALGORITHM,
            )
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/refresh", json={"refresh_token": token_no_jti}
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        assert "access_token" in r.json()


# ══ Auth — POST /auth/revoke ═════════════════════════════════════════════════════

class TestRevokeToken:
    def test_success_inserts_revocation_record(self) -> None:
        async def _run():
            session_local = await _make_session()
            s = get_settings()
            refresh_tok, _ = _make_token(
                "site_1", "client", timedelta(days=7), s.secret_key
            )
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/revoke", json={"refresh_token": refresh_tok}
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        assert r.json() == {"status": "revoked"}

    def test_invalid_jwt_still_returns_revoked(self) -> None:
        """Bad/expired token: JWTError is swallowed; endpoint always returns revoked."""
        async def _run():
            session_local = await _make_session()
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/revoke", json={"refresh_token": "garbage.token.value"}
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        assert r.json() == {"status": "revoked"}

    def test_already_revoked_is_idempotent(self) -> None:
        """Covers the `if existing.scalar_one_or_none() is None:` False branch."""
        async def _run():
            session_local = await _make_session()
            s = get_settings()
            refresh_tok, jti = _make_token(
                "site_1", "client", timedelta(days=7), s.secret_key
            )
            # Pre-populate so the jti already exists in the revocation table
            async with session_local() as sess:
                sess.add(RevokedToken(
                    jti=jti,
                    site_id="site_1",
                    expires_at=datetime.now(timezone.utc) + timedelta(days=7),
                ))
                await sess.commit()
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/revoke", json={"refresh_token": refresh_tok}
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        assert r.json() == {"status": "revoked"}

    def test_token_without_jti_skips_db_insert(self) -> None:
        """Covers the `if jti:` False branch inside revoke_token — no DB write attempted."""
        async def _run():
            session_local = await _make_session()
            s = get_settings()
            now = datetime.now(timezone.utc)
            token_no_jti = jwt.encode(
                {
                    "sub": "site_1",
                    "role": "client",
                    "iat": now,
                    "exp": now + timedelta(days=7),
                    # deliberately no "jti"
                },
                s.secret_key,
                algorithm=ALGORITHM,
            )
            app.dependency_overrides[get_db] = _db_override(session_local)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.post(
                        "/auth/revoke", json={"refresh_token": token_no_jti}
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        assert r.json() == {"status": "revoked"}
