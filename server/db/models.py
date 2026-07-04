"""
SQLAlchemy ORM models — the Python representation of database tables.

WHAT IS AN ORM?
---------------
ORM stands for Object-Relational Mapping. Instead of writing SQL directly
(e.g. `INSERT INTO site_registry VALUES (...)`), we define Python classes
whose attributes map to database columns. SQLAlchemy handles the SQL for us.

Benefits:
  - Type safety: Python type hints match column types
  - Portability: same code works with SQLite (dev) and PostgreSQL (prod)
  - Migrations: Alembic can compare these class definitions to the actual DB
    schema and generate the SQL needed to bring them in sync

IMPORTANT: Never alter the database schema manually (ALTER TABLE, etc.).
ALWAYS use Alembic migrations:
    alembic revision --autogenerate -m "describe your change"
    alembic upgrade head

PYTHON CONCEPT: Mapped[] and mapped_column()
  SQLAlchemy 2.x uses "mapped column" syntax:
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
  This declares both the Python type (int) and the DB column configuration.
  `Mapped[int]` = this column holds integers; `Mapped[str | None]` = nullable.

PYTHON CONCEPT: lambda for defaults
  `default=_now` passes the function _now as the default.
  SQLAlchemy calls `_now()` when creating a new row, so each row gets the
  current time, not the time the module was imported.
"""

from datetime import datetime, timezone   # datetime: timestamp; timezone: UTC
from sqlalchemy import DateTime, Index, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column
from server.db.database import Base      # Base is the SQLAlchemy declarative base


# A small helper that returns the current UTC time as a timezone-aware datetime.
# We define it once here so every column that needs "now" uses the same expression.
# The trailing `: lambda` type comment is a mypy/ruff workaround for bare lambdas.
_now = lambda: datetime.now(timezone.utc)  # noqa: E731


class SiteRegistry(Base):
    """
    Stores registered manufacturing sites and their hashed authentication secrets.

    This table is the "user database" of the system. There is one row per
    manufacturing site. The table is populated by `scripts/init_db.py` at
    startup and is not modified during normal FL operations.

    Columns
    -------
    id           : auto-incrementing integer primary key (internal use only)
    site_id      : the human-readable site name, e.g. "site_1" — must be unique
    secret_hash  : bcrypt hash of the site's secret — NEVER stored in plaintext
    registered_at: when this site was first registered (UTC, timezone-aware)
    last_seen    : when this site last successfully authenticated (updated on login)
                   Nullable because new sites haven't logged in yet.

    WHAT IS A PRIMARY KEY?
    ----------------------
    A primary key uniquely identifies each row. SQLAlchemy will automatically
    assign the next integer when a row is inserted.

    WHY STORE A HASH, NOT THE SECRET?
    ----------------------------------
    If the database were ever breached, an attacker would only find bcrypt hashes.
    Bcrypt hashes are computationally expensive to reverse — a brute-force attack
    against even a modest secret would take months. The actual secret is only
    known to the site operator who typed it into their `.env` file.
    """
    __tablename__ = "site_registry"   # the SQL table name

    id:            Mapped[int]            = mapped_column(Integer, primary_key=True)
    site_id:       Mapped[str]            = mapped_column(String(50), unique=True, nullable=False)
    secret_hash:   Mapped[str]            = mapped_column(String(256), nullable=False)
    registered_at: Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=_now)
    last_seen:     Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RoundRecord(Base):
    """
    Records one FL training round — its lifecycle and outcome.

    One row is created when a round starts and updated when it completes.
    The `global_metrics` column stores the aggregated performance numbers
    (average flux RMSE, LRV, etc.) for that round's global model.

    Columns
    -------
    id            : surrogate primary key
    round_id      : monotonically increasing round number (1, 2, 3, ...)
                    Unique — there is only one round with each number
    status        : lifecycle stage — see RoundStatus enum in schemas/federation.py
                    Values: "pending", "collecting", "aggregating", "complete", "failed"
    started_at    : when the round was initiated
    completed_at  : when aggregation finished; NULL while the round is still active
    model_version : the version number of the global model produced by this round
                    Version 0 = no model yet; increments by 1 each successful round
    global_metrics: JSON column storing a dict like:
                    {"flux_rmse": 1.18, "lrv_rmse": 0.05, "flux_ratio": 0.73}

    WHAT IS A JSON COLUMN?
    ----------------------
    PostgreSQL and SQLite both support storing arbitrary JSON objects in a single column.
    SQLAlchemy automatically serialises Python dicts to JSON on write and deserialises
    back to dicts on read. This avoids creating a separate table for round metrics.
    """
    __tablename__ = "rounds"

    id:             Mapped[int]            = mapped_column(Integer, primary_key=True)
    round_id:       Mapped[int]            = mapped_column(Integer, unique=True, nullable=False)
    status:         Mapped[str]            = mapped_column(String(20), default="pending")
    started_at:     Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=_now)
    completed_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    model_version:  Mapped[int]            = mapped_column(Integer, default=0)
    global_metrics: Mapped[dict]           = mapped_column(JSON, default=dict)


class ModelUpdateRecord(Base):
    """
    Records that a site submitted a model update for a specific round.

    IMPORTANT: The actual gradient tensors (delta_W) are NOT stored here.
    They are held in memory during the collection phase and discarded after
    aggregation. This is intentional — gradients can be large, and there is
    no regulatory requirement to persist them long-term.

    What IS stored:
      - Which site submitted (site_id)
      - Which round it was for (round_id)
      - How many training samples were used (n_samples) — needed for FedProx weighting
      - Which Hermia model was best for this run (hermia_model)
      - Summary performance metrics (local_metrics JSON)
      - When the update arrived (submitted_at)

    This audit trail is sufficient for compliance purposes without storing
    potentially sensitive gradient information.

    Columns
    -------
    site_id       : which site sent this update, e.g. "site_2"
    round_id      : the FL round this update belongs to (logical FK to rounds.round_id)
    n_samples     : local dataset size — used as the weight in FedProx averaging
    hermia_model  : which model had the lowest AIC: "standard", "complete", etc.
    local_metrics : JSON: {"flux_rmse": 1.23, "flux_ratio": 0.71, "amin_m2": 0.004}
    submitted_at  : when the server received this update
    """
    __tablename__ = "model_updates"

    id:            Mapped[int]     = mapped_column(Integer, primary_key=True)
    site_id:       Mapped[str]     = mapped_column(String(50), nullable=False)
    round_id:      Mapped[int]     = mapped_column(Integer, nullable=False)
    n_samples:     Mapped[int]     = mapped_column(Integer, nullable=False)
    hermia_model:  Mapped[str]     = mapped_column(String(30), default="combined_1a")
    local_metrics: Mapped[dict]    = mapped_column(JSON, default=dict)
    submitted_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RevokedToken(Base):
    """
    Persisted refresh-token revocation list.

    WHAT IS TOKEN REVOCATION?
    --------------------------
    JWTs (JSON Web Tokens) are stateless — the server doesn't store them.
    Normally, a token is valid until it expires. But if a site logs out
    (or a token is compromised), we want to invalidate it immediately.

    We do this by storing the token's unique identifier (jti = JWT ID) in
    this table when the token is consumed or explicitly revoked. On every
    token refresh, the server checks this table before issuing new tokens.

    HOW REFRESH TOKEN REVOCATION WORKS
    ------------------------------------
    1. Site calls POST /auth/refresh with refresh_token T1
    2. Server decodes T1, extracts jti = "abc123"
    3. Server checks: is "abc123" in revoked_tokens? No → token is valid
    4. Server INSERT INTO revoked_tokens (jti="abc123", ...)  ← T1 is now consumed
    5. Server issues new token pair T2, T3
    6. If someone later tries to use T1 again: step 3 returns YES → 401 Unauthorized

    Columns
    -------
    jti         : JWT ID — a UUID4 hex string (32 chars), unique, indexed for fast lookup
    site_id     : which site owned this token — for audit logging
    expires_at  : when the token was scheduled to expire (from its own exp claim)
                  Rows with expires_at < now() are safe to delete (periodic cleanup)
    revoked_at  : when the token was revoked or consumed

    WHY AN INDEX ON jti?
    --------------------
    Every /auth/refresh call queries this table by jti. Without an index, the DB
    would scan every row. With the index, the lookup is O(log n) — essentially
    instant even with millions of rows.

    `__table_args__` is how SQLAlchemy lets you add table-level constraints
    and indexes that don't belong on a single column.
    """
    __tablename__ = "revoked_tokens"

    id:         Mapped[int]      = mapped_column(Integer, primary_key=True)
    jti:        Mapped[str]      = mapped_column(String(64), unique=True, nullable=False)
    site_id:    Mapped[str]      = mapped_column(String(50), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Index on jti: makes the SELECT WHERE jti=? query very fast
    __table_args__ = (Index("ix_revoked_tokens_jti", "jti"),)
