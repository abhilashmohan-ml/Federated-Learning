"""
Global model REST endpoint — allows sites to download the current global weights.

PURPOSE
-------
After each FL round completes, the server has a new version of the global
model (the aggregated weights). Sites need to download these weights to
use them as the starting point for the next round of local training.

ENDPOINT
--------
GET /models/global-model

This returns the complete weight dictionary for the current global model.
The weights are a dict mapping layer names to lists of floats:
    {
        "hermia_params": [42.3, 0.012, 0.0003],
        ...
    }

If no rounds have completed yet (the server just started), this returns
a descriptive message rather than an error, because "no model yet" is a
normal state at startup.

SECURITY NOTE
-------------
This endpoint requires authentication (Bearer JWT) via get_current_site.
In a production system with more sophisticated privacy requirements, you
might also add per-site access controls or rate limiting here.

PYTHON CONCEPT: `if not weights:`
  An empty dict evaluates to False in Python, so `if not weights:` is True
  when the global model has no entries. This is a Pythonic way to check for
  "empty container" without comparing to `== {}`.
"""
from fastapi import APIRouter, Depends

from server.api.auth import get_current_site
from server.core.round_manager import RoundManager, get_round_manager

router = APIRouter()


@router.get("/global-model")
async def get_global_model(
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),   # auth gate — verifies JWT
) -> dict:
    """
    GET /models/global-model — download the current global model weights.

    Called by manufacturing sites at the start of each local training cycle.
    The site downloads these weights, uses them as the initial model state,
    trains on local data for LOCAL_EPOCHS, then submits the weight delta.

    Returns
    -------
    dict
        If a global model exists:
            {"layer_name": [float, ...], ...}  — the full weight dictionary
        If no rounds have completed yet:
            {"message": "No global model available yet"}
    """
    weights = rm.current_global_weights   # dict of layer → list[float]

    if not weights:
        # Normal at startup — no rounds have completed yet
        return {"message": "No global model available yet"}

    # Return the weights dict directly — FastAPI serialises it to JSON
    return weights
