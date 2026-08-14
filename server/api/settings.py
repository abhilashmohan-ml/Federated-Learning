"""GET/PUT /settings — runtime aggregation policy configuration."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from server.api.auth import get_current_site
from server.core.aggregation_policy import QuorumPolicy, TimeWindowPolicy
from server.core.round_manager import RoundManager, get_round_manager
from server.db.database import get_db
from server.db.settings_store import SettingsStore

router = APIRouter()


@router.get("")
async def get_settings(
    db: AsyncSession = Depends(get_db),
    _site: str = Depends(get_current_site),
) -> dict[str, Any]:
    """Return current aggregation policy configuration."""
    return await SettingsStore().load(db)


@router.put("")
async def update_settings(
    payload: dict[str, Any],
    db: AsyncSession = Depends(get_db),
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),
) -> dict[str, Any]:
    """
    Update one or more settings keys and apply the new policy live.

    Accepted keys: aggregation_mode ("quorum"|"time_window"),
                   quorum_min_sites (int str), time_window_seconds (int str),
                   heartbeat_seconds (int str).
    """
    store = SettingsStore()
    for key, value in payload.items():
        await store.save(db, key, str(value))
    await db.commit()

    config = await store.load(db)
    mode = config.get("aggregation_mode", "quorum")
    if mode == "time_window":
        rm.set_policy(TimeWindowPolicy(window_seconds=int(config["time_window_seconds"])))
    else:
        rm.set_policy(QuorumPolicy(min_sites=int(config["quorum_min_sites"])))

    return {"status": "ok", "config": config}
