"""Global model REST endpoints."""
from fastapi import APIRouter, Depends

from server.api.auth import get_current_site
from server.core.round_manager import RoundManager, get_round_manager

router = APIRouter()


@router.get("/global-model")
async def get_global_model(
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),
) -> dict:
    weights = rm.current_global_weights
    if not weights:
        return {"message": "No global model available yet"}
    return weights
