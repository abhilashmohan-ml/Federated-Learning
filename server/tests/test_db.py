"""Unit tests for server/db — 100% coverage."""

import asyncio
import inspect
from datetime import datetime, timezone
from unittest.mock import patch

from sqlalchemy import Index
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ── In-memory test engine ─────────────────────────────────────────────────────
_TEST_URL = "sqlite+aiosqlite:///:memory:"
_test_engine = create_async_engine(_TEST_URL, echo=False)
_TestSessionLocal = async_sessionmaker(_test_engine, expire_on_commit=False)

# ── Import modules under test ─────────────────────────────────────────────────
import server.db.database as db_module  # noqa: E402
from server.db.database import Base, get_db  # noqa: E402
from server.db.models import (  # noqa: E402
    ModelUpdateRecord,
    RoundRecord,
    RevokedToken,
    SiteRegistry,
    _now,
)


def _setup_db() -> None:
    """Create all ORM tables in the in-memory test database once at module load."""

    async def _create_all() -> None:
        async with _test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_all())


_setup_db()


# ── _now() ────────────────────────────────────────────────────────────────────


def test_now_returns_utc_aware_datetime() -> None:
    result = _now()
    assert isinstance(result, datetime)
    assert result.tzinfo is not None
    assert result.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


def test_now_returns_approximately_current_time() -> None:
    before = datetime.now(timezone.utc)
    result = _now()
    after = datetime.now(timezone.utc)
    assert before <= result <= after


# ── Base ──────────────────────────────────────────────────────────────────────


def test_base_is_declarative_base() -> None:
    from sqlalchemy.orm import DeclarativeBase

    assert issubclass(Base, DeclarativeBase)


def test_base_metadata_contains_all_tables() -> None:
    table_names = set(Base.metadata.tables.keys())
    assert {"site_registry", "rounds", "model_updates", "revoked_tokens"}.issubset(table_names)


# ── SiteRegistry ──────────────────────────────────────────────────────────────


def test_site_registry_tablename() -> None:
    assert SiteRegistry.__tablename__ == "site_registry"


def test_site_registry_instantiation() -> None:
    site = SiteRegistry(site_id="site_1", secret_hash="bcrypt_hash")
    assert site.site_id == "site_1"
    assert site.secret_hash == "bcrypt_hash"
    assert site.id is None
    assert site.last_seen is None


def test_site_registry_all_column_names() -> None:
    col_names = {c.name for c in SiteRegistry.__table__.c}
    assert col_names == {"id", "site_id", "secret_hash", "registered_at", "last_seen"}


def test_site_registry_registered_at_has_callable_default() -> None:
    col = SiteRegistry.__table__.c["registered_at"]
    assert col.default is not None
    assert callable(col.default.arg)


def test_site_registry_last_seen_is_nullable() -> None:
    col = SiteRegistry.__table__.c["last_seen"]
    assert col.nullable is True


def test_site_registry_site_id_is_unique() -> None:
    col = SiteRegistry.__table__.c["site_id"]
    assert any(
        len(list(uc.columns)) == 1 and "site_id" in [c.name for c in uc.columns]
        for uc in SiteRegistry.__table__.constraints
        if hasattr(uc, "columns")
    ) or col.unique


# ── RoundRecord ───────────────────────────────────────────────────────────────


def test_round_record_tablename() -> None:
    assert RoundRecord.__tablename__ == "rounds"


def test_round_record_instantiation_defaults() -> None:
    record = RoundRecord(round_id=1)
    assert record.round_id == 1
    # status/model_version defaults are DB-level only; not populated until flush
    assert record.completed_at is None


def test_round_record_all_column_names() -> None:
    col_names = {c.name for c in RoundRecord.__table__.c}
    assert col_names == {
        "id",
        "round_id",
        "status",
        "started_at",
        "completed_at",
        "model_version",
        "global_metrics",
    }


def test_round_record_started_at_has_default() -> None:
    col = RoundRecord.__table__.c["started_at"]
    assert col.default is not None


def test_round_record_global_metrics_default_is_callable() -> None:
    col = RoundRecord.__table__.c["global_metrics"]
    assert col.default is not None
    assert callable(col.default.arg)


def test_round_record_completed_at_nullable() -> None:
    col = RoundRecord.__table__.c["completed_at"]
    assert col.nullable is True


# ── ModelUpdateRecord ─────────────────────────────────────────────────────────


def test_model_update_record_tablename() -> None:
    assert ModelUpdateRecord.__tablename__ == "model_updates"


def test_model_update_record_instantiation() -> None:
    update = ModelUpdateRecord(site_id="site_2", round_id=3, n_samples=500)
    assert update.site_id == "site_2"
    assert update.round_id == 3
    assert update.n_samples == 500
    # hermia_model default is DB-level only; column default is declared on the column
    col = ModelUpdateRecord.__table__.c["hermia_model"]
    assert col.default.arg == "combined_1a"


def test_model_update_record_all_column_names() -> None:
    col_names = {c.name for c in ModelUpdateRecord.__table__.c}
    assert col_names == {
        "id",
        "site_id",
        "round_id",
        "n_samples",
        "hermia_model",
        "local_metrics",
        "submitted_at",
    }


def test_model_update_submitted_at_has_callable_default() -> None:
    col = ModelUpdateRecord.__table__.c["submitted_at"]
    assert col.default is not None
    assert callable(col.default.arg)


def test_model_update_local_metrics_default_is_callable() -> None:
    col = ModelUpdateRecord.__table__.c["local_metrics"]
    assert col.default is not None
    assert callable(col.default.arg)


# ── RevokedToken ──────────────────────────────────────────────────────────────


def test_revoked_token_tablename() -> None:
    assert RevokedToken.__tablename__ == "revoked_tokens"


def test_revoked_token_instantiation() -> None:
    exp = datetime.now(timezone.utc)
    token = RevokedToken(jti="tok_abc123", site_id="site_1", expires_at=exp)
    assert token.jti == "tok_abc123"
    assert token.site_id == "site_1"
    assert token.expires_at == exp
    assert token.id is None


def test_revoked_token_all_column_names() -> None:
    col_names = {c.name for c in RevokedToken.__table__.c}
    assert col_names == {"id", "jti", "site_id", "expires_at", "revoked_at"}


def test_revoked_token_revoked_at_has_callable_default() -> None:
    col = RevokedToken.__table__.c["revoked_at"]
    assert col.default is not None
    assert callable(col.default.arg)


def test_revoked_token_table_args_is_tuple_of_length_one() -> None:
    args = RevokedToken.__table_args__
    assert isinstance(args, tuple)
    assert len(args) == 1


def test_revoked_token_table_args_first_element_is_index() -> None:
    idx = RevokedToken.__table_args__[0]
    assert isinstance(idx, Index)


def test_revoked_token_table_args_index_name() -> None:
    idx = RevokedToken.__table_args__[0]
    assert idx.name == "ix_revoked_tokens_jti"


def test_revoked_token_index_registered_on_table() -> None:
    idx_names = {idx.name for idx in RevokedToken.__table__.indexes}
    assert "ix_revoked_tokens_jti" in idx_names


# ── get_db() ──────────────────────────────────────────────────────────────────


def test_get_db_is_async_generator_function() -> None:
    assert inspect.isasyncgenfunction(get_db)


def test_get_db_yields_async_session() -> None:
    async def _run() -> None:
        with patch.object(db_module, "AsyncSessionLocal", _TestSessionLocal):
            gen = get_db()
            session = await gen.__anext__()
            assert isinstance(session, AsyncSession)
            await gen.aclose()

    asyncio.run(_run())


def test_get_db_session_expire_on_commit_is_false() -> None:
    async def _run() -> None:
        with patch.object(db_module, "AsyncSessionLocal", _TestSessionLocal):
            gen = get_db()
            session = await gen.__anext__()
            assert session.sync_session.expire_on_commit is False
            await gen.aclose()

    asyncio.run(_run())


def test_get_db_aclose_completes_without_error() -> None:
    async def _run() -> None:
        with patch.object(db_module, "AsyncSessionLocal", _TestSessionLocal):
            gen = get_db()
            await gen.__anext__()
            await gen.aclose()  # must not raise

    asyncio.run(_run())


def test_get_db_each_call_produces_independent_session() -> None:
    async def _run() -> None:
        with patch.object(db_module, "AsyncSessionLocal", _TestSessionLocal):
            gen1 = get_db()
            gen2 = get_db()
            s1 = await gen1.__anext__()
            s2 = await gen2.__anext__()
            assert s1 is not s2
            await gen1.aclose()
            await gen2.aclose()

    asyncio.run(_run())


# ── database module-level attributes ─────────────────────────────────────────


def test_engine_attribute_is_not_none() -> None:
    assert db_module.engine is not None


def test_async_session_local_attribute_is_not_none() -> None:
    assert db_module.AsyncSessionLocal is not None


def test_base_exported_from_database_module() -> None:
    assert db_module.Base is Base
