"""Server configuration via pydantic-settings (reads .env)."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    secret_key: str = "CHANGE_ME"
    db_url: str = "sqlite+aiosqlite:///./viral_fl.db"
    host: str = "0.0.0.0"
    port: int = 8000
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


@lru_cache
def get_settings() -> ServerSettings:
    return ServerSettings()
