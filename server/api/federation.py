"""
Federation protocol REST endpoints.

OVERVIEW OF THE FL PROTOCOL
-----------------------------
The server exposes four endpoints that implement the federated learning
round lifecycle:

  1. POST /federation/round/start
     A site (or a scheduled job) calls this to kick off a new FL round.
     The server creates a new round in COLLECTING state and starts a timeout
     timer — if not enough sites respond, it aggregates whatever it has.

  2. POST /federation/update
     Each site calls this after completing local training, sending its
     model update (delta_W, n_samples, metrics). When min_sites_per_round
     updates arrive, aggregation is triggered automatically.

  3. GET /federation/round/{round_id}
     Any site can poll this to check the round's current status
     (COLLECTING, AGGREGATING, COMPLETE, FAILED).

  4. GET /federation/sites
     Returns the last-known status of each manufacturing site
     (IDLE, TRAINING, DONE).

AUTHENTICATION REQUIRED
------------------------
All these endpoints require a valid JWT. The `get_current_site` dependency
(from server/api/auth.py) acts as a gate — if the Authorization header is
missing or the token is invalid, FastAPI returns HTTP 401 before the
endpoint function even runs.

The `_site: str = Depends(get_current_site)` pattern:
  - The underscore prefix _ signals "we need this dependency for its
    side effect (authentication), but we don't use the return value."
  - For /update, we DO use the return value to verify that the site_id
    in the request body matches the authenticated site_id.

PYTHON CONCEPT: Dependency injection via Depends()
  RoundManager is a complex stateful object (it holds round history,
  update buffers, etc.). Instead of creating a new one per request,
  we use `get_round_manager()` which returns a cached singleton —
  the same RoundManager instance handles all requests for the server's
  lifetime. FastAPI injects it via `Depends(get_round_manager)`.

PYTHON CONCEPT: response_model=
  FastAPI uses this to validate and serialise the return value into JSON.
  If the endpoint returns a dict with unexpected fields, those fields are
  stripped. If required fields are missing, FastAPI raises an error.
  This gives automatic API contract enforcement.
"""
from fastapi import APIRouter, Depends, HTTPException

from shared.schemas.federation import FederationRound, ModelUpdate
from server.api.auth import get_current_site
from server.core.round_manager import RoundManager, get_round_manager

router = APIRouter()


@router.post("/round/start", response_model=FederationRound)
async def start_round(
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),   # auth gate — result unused
) -> FederationRound:
    """
    POST /federation/round/start — initiate a new FL round.

    Any authenticated site can start a round. In production this would
    typically be called by the server operator or a scheduled task.

    The returned FederationRound object contains:
      - round_id  : the numeric ID of the new round (1, 2, 3, ...)
      - status    : "collecting" — waiting for site updates
      - started_at: timestamp of when the round began
      - global_model_version: the model version sites should start from

    The RoundManager also starts an asyncio background task (timeout guard)
    that will force aggregation after `round_timeout_seconds` even if fewer
    than `min_sites_per_round` sites have responded.
    """
    return await rm.start_new_round()


@router.post("/update")
async def receive_update(
    update: ModelUpdate,
    rm: RoundManager = Depends(get_round_manager),
    site_id: str = Depends(get_current_site),   # authenticated site_id
) -> dict:
    """
    POST /federation/update — receive a model update from a manufacturing site.

    SECURITY: Compare the site_id from the JWT token with the site_id in the
    request body. A site cannot submit updates on behalf of another site.
    If they mismatch, we return HTTP 403 Forbidden.

    WHY THIS CHECK?
    ---------------
    A JWT proves "this request came from site_1". If the request body says
    `site_id: site_3`, something is wrong — either a bug or an attack attempt.
    The check prevents a compromised site from poisoning another site's updates.

    Parameters
    ----------
    update  : ModelUpdate — the gradient update, sample count, and metrics
    site_id : str — the site_id extracted from the JWT (authenticated identity)

    Returns
    -------
    dict — confirmation with site_id, round_id, and "accepted" status
    """
    # Ensure the authenticated identity matches the claimed identity
    if update.site_id != site_id:
        raise HTTPException(status_code=403, detail="site_id mismatch with token")

    # Hand the update to the round manager, which buffers it and triggers
    # aggregation if enough sites have now responded.
    await rm.receive_update(update)
    return {
        "status":   "accepted",
        "site_id":  update.site_id,
        "round_id": update.round_id,
    }


@router.get("/round/{round_id}", response_model=FederationRound)
async def get_round(
    round_id: int,
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),
) -> FederationRound:
    """
    GET /federation/round/{round_id} — query the status of a specific round.

    Sites poll this endpoint to find out:
      - Is the current round still COLLECTING (should we submit our update)?
      - Has the round COMPLETED (can we download the new global model)?
      - Did the round FAIL (should we wait for the next round)?

    PYTHON CONCEPT: path parameter
      The `{round_id}` in the URL is a path parameter. FastAPI extracts it
      and passes it as the `round_id: int` argument. It also validates the
      type — requesting /federation/round/abc returns 422 (invalid type).

    Returns
    -------
    FederationRound — round details, or HTTP 404 if that round doesn't exist
    """
    r = await rm.get_round(round_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Round not found")
    return r


@router.get("/sites")
async def list_sites(
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),
) -> dict:
    """
    GET /federation/sites — list the last-known status of all 5 manufacturing sites.

    Used by the server dashboard UI to show which sites are:
      - IDLE    : not currently participating in a round
      - TRAINING: actively running local training
      - DONE    : submitted their update for the current round

    Returns
    -------
    dict — {"sites": {"site_1": "idle", "site_2": "done", ...}}
    """
    return {"sites": await rm.get_site_statuses()}
