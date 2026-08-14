"""Async CRUD helper for the server_settings key-value table."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.models import ServerSetting

DEFAULTS: dict[str, str] = {
    "aggregation_mode": "quorum",
    "quorum_min_sites": "3",
    "time_window_seconds": "1800",
    "heartbeat_seconds": "30",
}


class SettingsStore:
    """Load and persist server runtime settings from/to the server_settings table."""

    async def load(self, db: AsyncSession) -> dict[str, str]:
        """Return all settings, falling back to DEFAULTS for missing keys."""
        result = await db.execute(select(ServerSetting))
        config: dict[str, str] = dict(DEFAULTS)
        for row in result.scalars().all():
            config[row.key] = row.value
        return config

    async def save(self, db: AsyncSession, key: str, value: str) -> None:
        """Insert or update a single setting. Caller must commit the session."""
        result = await db.execute(select(ServerSetting).where(ServerSetting.key == key))
        row = result.scalar_one_or_none()
        if row is None:
            db.add(ServerSetting(key=key, value=value))
        else:
            row.value = value
