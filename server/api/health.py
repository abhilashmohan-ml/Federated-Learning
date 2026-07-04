"""Health and readiness endpoints."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/metrics")
async def metrics() -> dict:
    # TODO: hook into Prometheus / round_manager counters
    return {"rounds_completed": 0, "sites_connected": 0}
