"""
Server configuration — reads environment variables and .env files.

HOW CONFIGURATION WORKS IN THIS PROJECT
-----------------------------------------
Rather than scattering hard-coded values across the codebase, ALL tunable
settings live here in ONE place. The settings are read from environment
variables at startup time (or from a .env file during development).

PYTHON CONCEPT: pydantic-settings (BaseSettings)
  BaseSettings is like Pydantic's BaseModel, but it automatically reads
  its field values from environment variables. For example, if your class
  has `port: int = 8000` and you set `SERVER_PORT=9000` in your .env file,
  the setting will be 9000.

  The mapping rule: field name → env var name (case-insensitive by default).
    secret_key   → SERVER_SECRET_KEY (or SECRET_KEY depending on env_prefix)
  For this project the env vars are named directly (no prefix required).

PYTHON CONCEPT: @lru_cache
  lru_cache = Least Recently Used cache. It memoises (remembers) the result
  of a function call so it only runs once.
  `get_settings()` creates a ServerSettings object — we call it from many
  different modules, but `@lru_cache` ensures the object is built exactly once
  and the same object is returned on every subsequent call.
  This also means all code shares the same settings instance.

SECURITY NOTES
--------------
- `secret_key` defaults to "CHANGE_ME" — this MUST be overridden in production
  via the SERVER_SECRET_KEY environment variable. If not changed, JWT tokens
  can be forged by anyone who knows this default.
- `ssl_keyfile` and `ssl_certfile` are None by default — leave them unset for
  HTTP (dev) and set them to enable HTTPS (prod). See docs/PRODUCTION.md.
"""
from functools import lru_cache   # Python standard library: function result cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Default CORS origins cover the Flet UI running locally.
# CORS = Cross-Origin Resource Sharing — browser security mechanism that controls
# which web origins (domains) are allowed to make requests to this server.
# The ports 8550-8555 map to: server dashboard (8550), sites 1-5 (8551-8555).
_DEFAULT_CORS = [
    "http://localhost:8550",
    "http://localhost:8551",
    "http://localhost:8552",
    "http://localhost:8553",
    "http://localhost:8554",
    "http://localhost:8555",
]


class ServerSettings(BaseSettings):
    """
    All server configuration in one class.

    ADDING A NEW SETTING
    --------------------
    1. Add a field here: `my_field: type = default_value`
    2. Add the corresponding line to .env.example
    3. Set the actual value in your .env file or Docker environment

    The `model_config` line tells pydantic-settings:
      - env_file=".env" : also read from a .env file (for development convenience)
      - extra="ignore"  : silently ignore env vars that don't match any field
                          (prevents errors if you have unrelated vars in your env)
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # JWT signing secret — CHANGE THIS in production!
    # Used to sign and verify JWT access/refresh tokens.
    # A compromised secret_key allows forging valid tokens for any site.
    secret_key: str = "CHANGE_ME"

    # Database connection string.
    # sqlite+aiosqlite = SQLite with async driver (fine for dev, no separate DB server needed)
    # postgresql+asyncpg = PostgreSQL with async driver (required for production)
    db_url: str = "sqlite+aiosqlite:///./viral_fl.db"

    host: str = "0.0.0.0"   # listen on all network interfaces (required inside Docker)
    port: int = 8000          # the FastAPI/uvicorn HTTP port

    # CORS_ORIGINS env var: comma-separated list of allowed web origins.
    # Leave empty to allow all origins without credentials (convenient for dev).
    # In production: set to the exact HTTPS URLs of your Flet UI instances.
    # Example: "https://server.example.com,https://site1.example.com"
    cors_origins: list[str] = _DEFAULT_CORS

    # TLS/HTTPS: provide both to enable encrypted connections.
    # If either is None (the default), the server runs on plain HTTP.
    # See scripts/generate_certs.sh for self-signed dev certs.
    ssl_keyfile: str | None  = None
    ssl_certfile: str | None = None

    flet_port: int  = 8550      # port for the server's Flet dashboard UI

    log_level: str  = "INFO"    # logging verbosity: DEBUG | INFO | WARNING | ERROR

    # FL protocol settings
    fl_rounds: int           = 50     # total number of FL rounds to run
    local_epochs: int        = 5      # training epochs per site per round
    learning_rate: float     = 0.001  # Adam optimiser learning rate
    fedprox_mu: float        = 0.01   # FedProx proximal term strength μ

    # Round management
    min_sites_per_round: int    = 3    # minimum sites needed before triggering aggregation
    round_timeout_seconds: int  = 300  # if not enough sites reply, aggregate anyway after this

    # JWT token lifetimes
    access_token_expire_minutes: int = 15    # short-lived: 15 minutes
    refresh_token_expire_days: int   = 7     # long-lived: 7 days

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        """
        Convert a comma-separated string into a Python list.

        Pydantic calls this validator before it tries to assign the value to
        the field. The `mode="before"` means it runs BEFORE type validation.

        Why needed? Environment variables are always strings. If CORS_ORIGINS
        is set to "https://a.com,https://b.com", we need to split it into
        ["https://a.com", "https://b.com"]. If the var is already a list
        (e.g. when set in Python tests), we pass it through unchanged.

        Parameters
        ----------
        v : the raw value from the environment (usually a str, but may be a list)

        Returns
        -------
        list[str] — parsed origins, or [] if the string was empty
        """
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                # Empty string → no explicit origins → dev mode (allow all)
                return []
            # Split on commas, strip whitespace from each item, drop empty strings
            return [o.strip() for o in stripped.split(",") if o.strip()]
        if isinstance(v, list):
            return v     # already a list — pass through
        return []         # unexpected type — safe default


@lru_cache
def get_settings() -> ServerSettings:
    """
    Return the singleton ServerSettings instance.

    The @lru_cache decorator ensures this function runs exactly once.
    All subsequent calls return the cached object without re-reading the env.

    Usage in FastAPI endpoints:
        settings: ServerSettings = Depends(get_settings)
    """
    return ServerSettings()
