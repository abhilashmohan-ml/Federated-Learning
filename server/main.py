"""
FastAPI application entry point for the FL aggregation server.

HOW THIS FILE IS STRUCTURED
-----------------------------
1. Create the FastAPI app object (`app`)
2. Add middleware (CORS handling)
3. Register all API routers under their URL prefixes
4. Start uvicorn when run directly with `python server/main.py`

PYTHON CONCEPT: FastAPI
  FastAPI is a modern Python web framework for building REST APIs.
  It automatically generates interactive API docs at /docs (Swagger UI)
  and /redoc. It uses Python type hints to validate request/response data
  via Pydantic.

PYTHON CONCEPT: Middleware
  Middleware is code that runs on EVERY request, before it reaches any
  endpoint handler. Think of it as a filter/wrapper around all routes.
  The CORSMiddleware here adds the appropriate HTTP headers to every
  response so browsers allow cross-origin requests.

PYTHON CONCEPT: Routers
  Instead of defining all endpoints in one huge file, FastAPI lets us
  split them into separate APIRouter objects (one per feature area) and
  then "include" them in the main app with a URL prefix:
    - auth endpoints   → /auth/...
    - federation       → /federation/...
    - model download   → /models/...
    - health checks    → /health/...

PYTHON CONCEPT: `if __name__ == "__main__":`
  This block runs ONLY when you execute the file directly:
    python server/main.py
  When another module imports this file (which uvicorn does), this block
  is NOT executed. It's the standard Python idiom for "run this as a script."

CORS EXPLAINED FOR BEGINNERS
------------------------------
When a browser (e.g. the Flet dashboard) at http://localhost:8550 makes
an HTTP request to http://localhost:8000 (this server), the browser blocks
it by default because the origins differ. The server must explicitly say
"I permit requests from http://localhost:8550" via CORS headers.

The tricky part: the CORS spec says you CANNOT use `allow_origins=["*"]`
(allow all) together with `allow_credentials=True`. Credentials (like
Authorization headers with JWT tokens) require explicit origins.

Our solution:
  - If cors_origins is empty (dev/no config) → allow all without credentials
    (works for quick testing; JWT still works because credentials are in the
    Authorization header, not cookies)
  - If cors_origins is set explicitly (production) → use that list WITH credentials
"""
import uvicorn                              # the ASGI server that runs FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api import auth, federation, models, health   # our API modules
from server.config import get_settings
from shared.utils.logging_config import configure_logging

# Configure structured logging before anything else
configure_logging()
settings = get_settings()

# Create the FastAPI application instance
app = FastAPI(
    title="Viral Filtration FL Server",
    version="0.1.0",
    description="Federated learning aggregation server for mAb viral filtration",
)

# ── CORS middleware ────────────────────────────────────────────────────────────
#
# `allow_credentials=True` with `allow_origins=["*"]` violates the CORS spec
# and raises a runtime error in Starlette. We handle the two modes separately.
if settings.cors_origins:
    # Production mode: explicit origin list with credentials allowed.
    # Bearer tokens sent in Authorization headers work correctly here.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,   # only these origins are permitted
        allow_credentials=True,                 # allow Authorization header
        allow_methods=["*"],                    # allow GET, POST, etc.
        allow_headers=["*"],                    # allow all headers
    )
else:
    # Development mode: all origins allowed, but credentials not exposed.
    # JWT tokens in Authorization headers still work (not affected by this).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],       # any origin may request
        allow_credentials=False,   # required when allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ── Register API routers ───────────────────────────────────────────────────────
#
# `prefix` prepends a path to every route in that router.
# `tags` groups the endpoints under a named section in the /docs UI.
app.include_router(auth.router,       prefix="/auth",       tags=["auth"])
app.include_router(federation.router, prefix="/federation", tags=["federation"])
app.include_router(models.router,     prefix="/models",     tags=["models"])
app.include_router(health.router,     prefix="/health",     tags=["health"])


if __name__ == "__main__":
    # Collect SSL kwargs only if both key and cert are configured.
    # **ssl_kw unpacks the dict as keyword arguments to uvicorn.run().
    # If ssl_kw is empty, no SSL args are passed → plain HTTP.
    ssl_kw: dict[str, str] = {}
    if settings.ssl_keyfile and settings.ssl_certfile:
        ssl_kw = {
            "ssl_keyfile": settings.ssl_keyfile,
            "ssl_certfile": settings.ssl_certfile,
        }

    # `"server.main:app"` tells uvicorn where to find the app object.
    # reload=False in production — reloading watches for file changes and
    # restarts automatically, which is useful in dev but unnecessary (and
    # slightly slower) in production Docker containers.
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        **ssl_kw,   # expands to ssl_keyfile=..., ssl_certfile=... if set
    )
