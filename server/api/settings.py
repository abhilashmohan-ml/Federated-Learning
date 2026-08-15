"""GET/PUT /settings — runtime aggregation policy configuration."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from server.api.auth import get_current_site, require_admin_token
from shared.utils.logging_config import get_logger
from server.core.aggregation_policy import QuorumPolicy, TimeWindowPolicy
from server.core.round_manager import RoundManager, get_round_manager
from server.db.database import get_db
from server.db.settings_store import SettingsStore

router = APIRouter()
log = get_logger(__name__)

# Complete set of keys the /settings API accepts.  Any other key is rejected
# with 422 so callers get a clear error instead of silently dirtying the DB.
_ALLOWED_KEYS: frozenset[str] = frozenset(
    {"aggregation_mode", "quorum_min_sites", "time_window_seconds", "heartbeat_seconds"}
)

# Subset of allowed keys whose values must parse as integers.
_NUMERIC_KEYS: frozenset[str] = frozenset(
    {"quorum_min_sites", "time_window_seconds", "heartbeat_seconds"}
)


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
    _: None = Depends(require_admin_token),
) -> dict[str, Any]:
    """
    Update one or more settings keys and apply the new policy live.

    Requires ``X-Admin-Key`` header matching the server's ``SECRET_KEY``.
    Regular site JWT tokens are rejected with HTTP 403.

    Accepted keys: aggregation_mode ("quorum"|"time_window"),
                   quorum_min_sites (int str), time_window_seconds (int str),
                   heartbeat_seconds (int str).
    """
    # Reject unknown keys before touching the database.
    unknown = set(payload) - _ALLOWED_KEYS
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown settings key(s): {', '.join(sorted(unknown))}",
        )

    # Validate numeric fields BEFORE touching the database.
    for key, value in payload.items():
        if key in _NUMERIC_KEYS:
            try:
                int(value)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid value for {key}: must be an integer",
                )

    store = SettingsStore()
    for key, value in payload.items():
        await store.save(db, key, str(value))
    await db.commit()

    config = await store.load(db)
    mode = config.get("aggregation_mode", "quorum")
    try:
        if mode == "time_window":
            rm.set_policy(TimeWindowPolicy(window_seconds=int(config["time_window_seconds"])))
        else:
            rm.set_policy(QuorumPolicy(min_sites=int(config["quorum_min_sites"])))
    except (ValueError, KeyError) as exc:
        log.warning("settings_apply_policy_failed_using_default", error=str(exc))
        rm.set_policy(QuorumPolicy())

    return {"status": "ok", "config": config}
