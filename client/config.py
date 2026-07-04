"""Client configuration via pydantic-settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClientSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    site_id: str            = "site_1"
    server_url: str         = "http://localhost:8000"
    site_secret: str        = "secret_site_1"
    dp_noise_sigma: float   = 0.01
    local_data_path: str    = "./data/site_1/filtration.csv"
    flet_client_port: int   = 8551
    log_level: str          = "INFO"
    local_epochs: int       = 5
    learning_rate: float    = 0.001
    fedprox_mu: float       = 0.01
    # Network resilience / security
    verify_ssl: bool        = True   # set False only with self-signed dev certs
    connect_timeout: int    = 10     # seconds to establish TCP connection
    request_timeout: int    = 60     # seconds to wait for response
    retry_attempts: int     = 3      # retries on transient network errors


@lru_cache
def get_client_settings() -> ClientSettings:
    return ClientSettings()
