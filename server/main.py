"""FastAPI application entry point for the FL aggregation server."""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api import auth, federation, models, health
from server.config import get_settings
from shared.utils.logging_config import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title="Viral Filtration FL Server",
    version="0.1.0",
    description="Federated learning aggregation server for mAb viral filtration",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,       prefix="/auth",       tags=["auth"])
app.include_router(federation.router, prefix="/federation", tags=["federation"])
app.include_router(models.router,     prefix="/models",     tags=["models"])
app.include_router(health.router,     prefix="/health",     tags=["health"])


if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
