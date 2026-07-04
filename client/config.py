"""
Client (manufacturing site) configuration — reads environment variables and .env.

SITE IDENTITY
--------------
Each of the 5 manufacturing sites runs an identical copy of the client code,
but with different environment variables. The SITE_ID and SITE_SECRET
variables are what distinguish site_1 from site_2, site_3, etc.

When running via Docker Compose, each site container gets its own SITE_ID
and SITE_SECRET from the .env file. When deploying to real servers, set
these as actual environment variables on the remote machine.

PYTHON CONCEPT: lru_cache
  `get_client_settings()` is cached so the settings object is created once
  and reused. All modules (data_loader, local_trainer, fl_client, heartbeat)
  call `get_client_settings()` but they all receive the SAME object.
  This is important because the cached object also holds the JWT tokens
  indirectly via the FLClient (which uses these settings).

PRIVACY NOTE
------------
`local_data_path` points to the CSV file on the local filesystem.
This path stays entirely local — the CSV file is NEVER uploaded to the server.
Only model updates (gradient deltas) leave the site container.
See client/engine/local_trainer.py for how the data is used.

NETWORK RESILIENCE SETTINGS
-----------------------------
`verify_ssl`:      Whether to verify the server's TLS certificate.
                   Set to False ONLY when using self-signed dev certs.
                   MUST be True in production — False allows MITM attacks.
`connect_timeout`: Seconds to wait for a TCP connection to be established.
                   If the server is unreachable (network partition, server down),
                   the connection attempt fails after this many seconds.
`request_timeout`: Seconds to wait for the server to send back a response.
                   Training can take a while, so set this higher than connect_timeout.
`retry_attempts`:  How many times to retry a failed request before giving up.
                   Retries use exponential backoff: 2s, 4s, 8s between attempts.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClientSettings(BaseSettings):
    """
    Configuration for a single manufacturing site client.

    All settings can be overridden via environment variables. For local dev,
    put them in a .env file in the project root. For Docker deployments, set
    them in docker-compose.yml or as actual container environment variables.

    Example .env for site_2:
        SITE_ID=site_2
        SITE_SECRET=my_very_secret_passphrase
        SERVER_URL=https://flserver.mycompany.com
        LOCAL_DATA_PATH=/data/filtration.csv
        DP_NOISE_SIGMA=0.005
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Site identity ───────────────────────────────────────────────────────────
    site_id:         str   = "site_1"           # unique name: site_1, site_2, ..., site_5
    server_url:      str   = "http://localhost:8000"  # URL of the FL aggregation server
    site_secret:     str   = "secret_site_1"    # authentication secret (CHANGE IN PRODUCTION)

    # ── Privacy ─────────────────────────────────────────────────────────────────
    dp_noise_sigma:  float = 0.01               # Gaussian DP noise σ added to gradients
                                                # Larger σ → more privacy, less accuracy

    # ── Data ────────────────────────────────────────────────────────────────────
    local_data_path: str   = "./data/site_1/filtration.csv"  # NEVER sent to server

    # ── UI ──────────────────────────────────────────────────────────────────────
    flet_client_port: int  = 8551   # web port for the Flet status dashboard

    # ── Logging ─────────────────────────────────────────────────────────────────
    log_level:       str   = "INFO"   # DEBUG | INFO | WARNING | ERROR

    # ── FL training hyperparameters ─────────────────────────────────────────────
    local_epochs:    int   = 5      # how many epochs to train locally each round
    learning_rate:   float = 0.001  # Adam optimiser step size
    fedprox_mu:      float = 0.01   # FedProx proximal term strength (see PINN loss)

    # ── Network resilience / security ───────────────────────────────────────────
    verify_ssl:      bool  = True   # MUST be True in production
    connect_timeout: int   = 10     # seconds to establish TCP connection
    request_timeout: int   = 60     # seconds to wait for server response
    retry_attempts:  int   = 3      # max retries on transient network errors


@lru_cache
def get_client_settings() -> ClientSettings:
    """
    Return the singleton ClientSettings object.

    All modules import and call this function. lru_cache ensures the .env file
    is read exactly once. This also means changing the .env file while the
    client is running has no effect until the process is restarted.
    """
    return ClientSettings()
