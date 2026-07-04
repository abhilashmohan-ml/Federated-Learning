"""Server configuration via pydantic-settings (reads .env)."""
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_CORS = [
    "http://localhost:8550",
    "http://localhost:8551",
    "http://localhost:8552",
    "http://localhost:8553",
    "http://localhost:8554",
    "http://localhost:8555",
]


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    secret_key: str = "CHANGE_ME"
    db_url: str = "sqlite+aiosqlite:///./viral_fl.db"
    host: str = "0.0.0.0"
    port: int = 8000
    # CORS_ORIGINS env var: comma-separated list of allowed origins.
    # Leave empty (default) to allow all origins without credentials (dev mode).
    # In production set to e.g. "https://site1.example.com,https://site2.example.com"
    cors_origins: list[str] = _DEFAULT_CORS
    # TLS: set both to enable HTTPS
    ssl_keyfile: str | None = None
    ssl_certfile: str | None = None
    flet_port: int = 8550
    log_level: str = "INFO"
    fl_rounds: int = 50
    local_epochs: int = 5
    learning_rate: float = 0.001
    fedprox_mu: float = 0.01
    min_sites_per_round: int = 3
    round_timeout_seconds: int = 300
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            return [o.strip() for o in stripped.split(",") if o.strip()]
        if isinstance(v, list):
            return v
        return []


@lru_cache
def get_settings() -> ServerSettings:
    return ServerSettings()
