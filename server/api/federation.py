"""
Federation protocol endpoints.  All routes require a valid Bearer JWT.

POST /federation/round/start   start a new FL round
POST /federation/update        receive model update from a site
GET  /federation/round/{id}    get round status
GET  /federation/sites         list all site statuses
"""
from fastapi import APIRouter, Depends, HTTPException

from shared.schemas.federation import FederationRound, ModelUpdate
from server.api.auth import get_current_site
from server.core.round_manager import RoundManager, get_round_manager

router = APIRouter()


@router.post("/round/start", response_model=FederationRound)
async def start_round(
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),
) -> FederationRound:
    return await rm.start_new_round()


@router.post("/update")
async def receive_update(
    update: ModelUpdate,
    rm: RoundManager = Depends(get_round_manager),
    site_id: str = Depends(get_current_site),
) -> dict:
    if update.site_id != site_id:
        raise HTTPException(status_code=403, detail="site_id mismatch with token")
    await rm.receive_update(update)
    return {"status": "accepted", "site_id": update.site_id, "round_id": update.round_id}


@router.get("/round/{round_id}", response_model=FederationRound)
async def get_round(
    round_id: int,
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),
) -> FederationRound:
    r = await rm.get_round(round_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Round not found")
    return r


@router.get("/sites")
async def list_sites(
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),
) -> dict:
    return {"sites": await rm.get_site_statuses()}
