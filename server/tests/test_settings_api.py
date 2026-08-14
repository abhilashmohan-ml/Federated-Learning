"""Tests for Settings API (GET/PUT /settings) and GET /federation/current-round."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from server.api.auth import _make_token
from server.config import get_settings
from server.core.round_manager import get_round_manager
from server.db.database import Base, get_db
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


# ── Auth header helpers ─────────────────────────────────────────────────────────

def _auth_headers(site_id: str = "site_1") -> dict:
    """Return Authorization headers with a valid short-lived JWT for *site_id*."""
    s = get_settings()
    token, _ = _make_token(site_id, "client", timedelta(minutes=15), s.secret_key)
    return {"Authorization": f"Bearer {token}"}


def _admin_headers() -> dict:
    """Return X-Admin-Key header matching the server's secret_key."""
    s = get_settings()
    return {"X-Admin-Key": s.secret_key}


# ── Mock RoundManager helper ────────────────────────────────────────────────────

def _fed_round(round_id: int = 1) -> FederationRound:
    return FederationRound(
        round_id=round_id,
        status=RoundStatus.COLLECTING,
        started_at=datetime.now(timezone.utc),
    )


def _mock_rm() -> MagicMock:
    """Return a MagicMock pre-configured to look like a RoundManager."""
    rm = MagicMock()
    rm.get_or_create_round = AsyncMock(return_value=_fed_round(1))
    rm.set_policy = MagicMock()
    return rm


# ══ GET /settings ════════════════════════════════════════════════════════════════

class TestGetSettings:
    def test_returns_defaults_when_no_db_rows(self) -> None:
        """GET /settings with empty DB returns the DEFAULTS dict."""
        async def _run():
            session_local = await _make_session()
            rm = _mock_rm()
            app.dependency_overrides[get_db] = _db_override(session_local)
            app.dependency_overrides[get_round_manager] = lambda: rm
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.get("/settings", headers=_auth_headers())
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        data = r.json()
        assert "aggregation_mode" in data
        assert data["aggregation_mode"] == "quorum"
        assert data["quorum_min_sites"] == "3"
        assert data["time_window_seconds"] == "1800"

    def test_requires_auth_returns_401(self) -> None:
        """GET /settings without a token returns 401."""
        async def _run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.get("/settings")  # no Authorization header

        r = asyncio.run(_run())
        assert r.status_code == 401


# ══ PUT /settings ════════════════════════════════════════════════════════════════

class TestPutSettings:
    def test_update_to_time_window_returns_ok(self) -> None:
        """PUT {"aggregation_mode": "time_window"} → {"status": "ok", "config": {...}}."""
        async def _run():
            session_local = await _make_session()
            rm = _mock_rm()
            app.dependency_overrides[get_db] = _db_override(session_local)
            app.dependency_overrides[get_round_manager] = lambda: rm
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.put(
                        "/settings",
                        headers=_admin_headers(),
                        json={"aggregation_mode": "time_window"},
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "config" in body
        assert body["config"]["aggregation_mode"] == "time_window"

    def test_update_to_quorum_returns_ok(self) -> None:
        """PUT {"aggregation_mode": "quorum"} → config has aggregation_mode=quorum."""
        async def _run():
            session_local = await _make_session()
            rm = _mock_rm()
            app.dependency_overrides[get_db] = _db_override(session_local)
            app.dependency_overrides[get_round_manager] = lambda: rm
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.put(
                        "/settings",
                        headers=_admin_headers(),
                        json={"aggregation_mode": "quorum"},
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["config"]["aggregation_mode"] == "quorum"

    def test_policy_set_called_on_round_manager(self) -> None:
        """PUT /settings calls rm.set_policy exactly once."""
        async def _run():
            session_local = await _make_session()
            rm = _mock_rm()
            app.dependency_overrides[get_db] = _db_override(session_local)
            app.dependency_overrides[get_round_manager] = lambda: rm
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    await client.put(
                        "/settings",
                        headers=_admin_headers(),
                        json={"aggregation_mode": "time_window"},
                    )
                return rm
            finally:
                app.dependency_overrides.clear()

        rm_after = asyncio.run(_run())
        rm_after.set_policy.assert_called_once()

    def test_requires_admin_key_returns_403(self) -> None:
        """PUT /settings without X-Admin-Key returns 403."""
        async def _run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.put(
                    "/settings", json={"aggregation_mode": "quorum"}
                )

        r = asyncio.run(_run())
        assert r.status_code == 403

    def test_put_settings_rejected_without_admin_auth(self) -> None:
        """PUT /settings with a valid site JWT but no X-Admin-Key returns 403."""
        async def _run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.put(
                    "/settings",
                    headers=_auth_headers(),   # valid client JWT, but no admin key
                    json={"aggregation_mode": "quorum"},
                )

        r = asyncio.run(_run())
        assert r.status_code == 403

    def test_put_settings_invalid_numeric_returns_422(self) -> None:
        """PUT /settings with a non-integer numeric field returns 422 before DB write."""
        async def _run():
            session_local = await _make_session()
            rm = _mock_rm()
            app.dependency_overrides[get_db] = _db_override(session_local)
            app.dependency_overrides[get_round_manager] = lambda: rm
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.put(
                        "/settings",
                        headers=_admin_headers(),
                        json={"quorum_min_sites": "not_a_number"},
                    )
            finally:
                app.dependency_overrides.clear()

        r = asyncio.run(_run())
        assert r.status_code == 422
        assert "Invalid value" in r.json()["detail"]


# ══ GET /federation/current-round ════════════════════════════════════════════════

class TestGetCurrentRound:
    def test_returns_round_with_round_id(self) -> None:
        """GET /federation/current-round returns a FederationRound JSON with round_id."""
        rm = _mock_rm()
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    return await client.get(
                        "/federation/current-round", headers=_auth_headers()
                    )

            r = asyncio.run(_run())
            assert r.status_code == 200
            body = r.json()
            assert "round_id" in body
            assert body["round_id"] == 1
            assert body["status"] == "collecting"
        finally:
            app.dependency_overrides.clear()

    def test_calls_get_or_create_round(self) -> None:
        """GET /federation/current-round delegates to rm.get_or_create_round()."""
        rm = _mock_rm()
        try:
            app.dependency_overrides[get_round_manager] = lambda: rm

            async def _run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    await client.get(
                        "/federation/current-round", headers=_auth_headers()
                    )
                return rm

            rm_after = asyncio.run(_run())
            rm_after.get_or_create_round.assert_awaited_once()
        finally:
            app.dependency_overrides.clear()

    def test_requires_auth_returns_401(self) -> None:
        """GET /federation/current-round without a token returns 401."""
        async def _run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.get("/federation/current-round")

        r = asyncio.run(_run())
        assert r.status_code == 401
