"""
Health and readiness endpoints — used by monitoring systems and load balancers.

WHY HEALTH ENDPOINTS?
----------------------
In production, infrastructure tools need a way to check whether a service is
running correctly:

  Kubernetes: probes /health/ every 10 seconds. If it returns anything other
              than 2xx, the container is restarted.
  Docker Compose: HEALTHCHECK directive polls this endpoint.
  Load balancers: only route traffic to healthy instances.
  Monitoring: Prometheus scrapes /health/metrics to track system state.

ENDPOINT: GET /health/
  The simplest possible check — returns {"status": "ok"} if the server
  process is alive and can handle requests. Does NOT check the database;
  that would be a "readiness probe" and is a separate concern.

ENDPOINT: GET /health/metrics
  Placeholder for aggregated operational metrics (round counters, site
  counts, etc.). Currently returns zeros. In a production system this would:
    - Query the RoundManager for active round count and completed rounds
    - Query the DB for connected site count
    - Expose Prometheus-format metrics (via prometheus-fastapi-instrumentator)

PYTHON CONCEPT: async def with no await
  These endpoints are declared `async` for consistency with FastAPI's
  async routing, but they don't actually await anything (no DB calls, no I/O).
  This is fine — FastAPI handles sync and async endpoints uniformly.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health() -> dict:
    """
    GET /health/ — simple liveness probe.

    Returns HTTP 200 with {"status": "ok"} if the server is running.
    This is the minimum viable health check — it confirms the server
    process is alive and the event loop is processing requests.

    No authentication required — health checks run from infrastructure
    components that don't hold JWT tokens.
    """
    return {"status": "ok"}


@router.get("/metrics")
async def metrics() -> dict:
    """
    GET /health/metrics — operational metrics for the monitoring dashboard.

    TODO: Connect to RoundManager and DB counters.
    Currently returns placeholder zeros. A future implementation would:
      - rm.get_completed_round_count() → rounds_completed
      - len(rm.get_connected_sites())  → sites_connected
      - rm.get_average_round_duration() → avg_round_time_s

    No authentication required — metrics are typically scraped by monitoring
    agents that run on the same internal network as the server.
    """
    return {
        "rounds_completed": 0,    # placeholder — connect to RoundManager
        "sites_connected":  0,    # placeholder — connect to heartbeat tracker
    }
