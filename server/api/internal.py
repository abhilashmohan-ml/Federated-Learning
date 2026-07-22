"""Internal read-only status endpoint — no authentication required."""
from fastapi import APIRouter, Depends

from server.core.round_manager import RoundManager, get_round_manager

router = APIRouter()


@router.get("/status")
async def get_internal_status(
    rm: RoundManager = Depends(get_round_manager),
) -> dict:
    """
    GET /internal/status — return current round and site state.

    No JWT required. This endpoint is read-only and is only called by the
    Flet dashboard running on the same host. It exposes no sensitive data
    (only round status and site training phase — no weights or secrets).
    """
    return await rm.get_status_snapshot()
