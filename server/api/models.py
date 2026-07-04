"""Global model REST endpoints."""
from fastapi import APIRouter
from shared.schemas.federation import GlobalModel

router = APIRouter()
_global_model: dict = {}


@router.get("/global-model")
async def get_global_model() -> dict:
    return _global_model or {"message": "No global model available yet"}


@router.post("/global-model")
async def set_global_model(model: GlobalModel) -> dict:
    global _global_model
    _global_model = model.model_dump()
    return {"status": "updated", "version": model.version}
