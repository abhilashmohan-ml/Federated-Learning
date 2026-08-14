"""Tests for server/db/settings_store.py — 100% coverage required."""
from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from server.db.database import Base
from server.db.settings_store import SettingsStore


@pytest_asyncio.fixture
async def db() -> AsyncSession:  # type: ignore[misc]
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_defaults_returned_when_table_empty(db: AsyncSession) -> None:
    store = SettingsStore()
    config = await store.load(db)
    assert config["aggregation_mode"] == "quorum"
    assert config["quorum_min_sites"] == "3"
    assert config["time_window_seconds"] == "1800"
    assert config["heartbeat_seconds"] == "30"


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(db: AsyncSession) -> None:
    store = SettingsStore()
    await store.save(db, "aggregation_mode", "time_window")
    await db.commit()
    config = await store.load(db)
    assert config["aggregation_mode"] == "time_window"


@pytest.mark.asyncio
async def test_update_existing_key(db: AsyncSession) -> None:
    store = SettingsStore()
    await store.save(db, "quorum_min_sites", "4")
    await db.commit()
    await store.save(db, "quorum_min_sites", "5")
    await db.commit()
    config = await store.load(db)
    assert config["quorum_min_sites"] == "5"


@pytest.mark.asyncio
async def test_unknown_key_not_in_defaults_stored_and_returned(db: AsyncSession) -> None:
    store = SettingsStore()
    await store.save(db, "custom_key", "custom_value")
    await db.commit()
    config = await store.load(db)
    assert config["custom_key"] == "custom_value"


@pytest.mark.asyncio
async def test_db_value_overrides_default(db: AsyncSession) -> None:
    store = SettingsStore()
    await store.save(db, "quorum_min_sites", "2")
    await db.commit()
    config = await store.load(db)
    assert config["quorum_min_sites"] == "2"  # DB wins over default "3"
