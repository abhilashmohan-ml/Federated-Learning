# Database Schema
## Viral Filtration Federated Learning Platform

**Version:** 1.0  
**Date:** 2026-07-04  
**ORM:** SQLAlchemy 2.x (asyncio) — `server/db/models.py`  
**Migrations:** Alembic — `server/db/migrations/`  
**Development DB:** SQLite (aiosqlite)  
**Production DB:** PostgreSQL 16

> **Rule:** Never alter the schema directly. All changes must go through an Alembic migration (`alembic revision --autogenerate -m "description"` then `alembic upgrade head`).

---

## Entity-Relationship Overview

```
site_registry ──(1:N)── model_updates
      │
      └──(1:N)── revoked_tokens

rounds ──(1:N)── model_updates  (by round_id, logical FK — not enforced in schema)
```

No foreign key constraints are currently defined at the DB level. Referential integrity is enforced at the application layer.

---

## Table: `site_registry`

Stores the registered manufacturing sites and their hashed authentication secrets.  
Populated by `scripts/init_db.py` at first startup.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTO | Surrogate key |
| `site_id` | VARCHAR(50) | UNIQUE, NOT NULL | Site identifier: `site_1` … `site_5` |
| `secret_hash` | VARCHAR(256) | NOT NULL | bcrypt hash of the site secret (passlib `CryptContext(schemes=["bcrypt"])`) |
| `registered_at` | TIMESTAMP WITH TIME ZONE | NOT NULL, DEFAULT now() | Registration timestamp |
| `last_seen` | TIMESTAMP WITH TIME ZONE | NULL | Last successful authentication timestamp |

**Indexes:**
- `PRIMARY KEY (id)`
- `UNIQUE (site_id)` — implicit from UNIQUE constraint

**Notes:**
- `secret_hash` is never logged and never returned by any API
- `last_seen` is updated on each successful `/auth/token` call (planned — not yet implemented in v0.1.0)
- bcrypt work factor: passlib default (12 rounds)

**SQLAlchemy definition:**
```python
class SiteRegistry(Base):
    __tablename__ = "site_registry"
    id:            Mapped[int]           = mapped_column(Integer, primary_key=True)
    site_id:       Mapped[str]           = mapped_column(String(50), unique=True, nullable=False)
    secret_hash:   Mapped[str]           = mapped_column(String(256), nullable=False)
    registered_at: Mapped[datetime]      = mapped_column(DateTime(timezone=True), default=_now)
    last_seen:     Mapped[datetime|None] = mapped_column(DateTime(timezone=True), nullable=True)
```

---

## Table: `rounds`

One row per federation round. Records the lifecycle and outcome of each FL round.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTO | Surrogate key |
| `round_id` | INTEGER | UNIQUE, NOT NULL | Monotonically incrementing round number (1-based) |
| `status` | VARCHAR(20) | NOT NULL, DEFAULT `'pending'` | Round state: `pending` \| `collecting` \| `aggregating` \| `complete` \| `failed` |
| `started_at` | TIMESTAMP WITH TIME ZONE | NOT NULL, DEFAULT now() | When the round was started |
| `completed_at` | TIMESTAMP WITH TIME ZONE | NULL | When aggregation completed (or failed) |
| `model_version` | INTEGER | NOT NULL, DEFAULT 0 | Global model version after this round |
| `global_metrics` | JSON | NOT NULL, DEFAULT `{}` | Aggregated metrics: `flux_rmse`, `lrv_rmse`, `flux_ratio`, `amin_m2` |

**Indexes:**
- `PRIMARY KEY (id)`
- `UNIQUE (round_id)`

**Status transitions:**
```
pending → collecting → aggregating → complete
                                  → failed
```

**`global_metrics` JSON structure:**
```json
{
  "flux_rmse": 1.18,
  "lrv_rmse": 0.05,
  "flux_ratio": 0.73,
  "amin_m2": 0.0040
}
```

**SQLAlchemy definition:**
```python
class RoundRecord(Base):
    __tablename__ = "rounds"
    id:            Mapped[int]           = mapped_column(Integer, primary_key=True)
    round_id:      Mapped[int]           = mapped_column(Integer, unique=True, nullable=False)
    status:        Mapped[str]           = mapped_column(String(20), default="pending")
    started_at:    Mapped[datetime]      = mapped_column(DateTime(timezone=True), default=_now)
    completed_at:  Mapped[datetime|None] = mapped_column(DateTime(timezone=True), nullable=True)
    model_version: Mapped[int]           = mapped_column(Integer, default=0)
    global_metrics:Mapped[dict]          = mapped_column(JSON, default=dict)
```

---

## Table: `model_updates`

One row per model update submitted by a site. Records what was received and when; does not store the actual gradient tensors (those are aggregated in memory and discarded).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTO | Surrogate key |
| `site_id` | VARCHAR(50) | NOT NULL | Submitting site (`site_1` … `site_5`) |
| `round_id` | INTEGER | NOT NULL | Target round (logical FK → `rounds.round_id`) |
| `n_samples` | INTEGER | NOT NULL | Number of local training samples; used for FedProx weighting |
| `hermia_model` | VARCHAR(30) | NOT NULL, DEFAULT `'combined_1a'` | Best Hermia model selected by AIC: `standard` \| `complete` \| `intermediate` \| `cake` \| `combined_1a` |
| `local_metrics` | JSON | NOT NULL, DEFAULT `{}` | Per-site metrics: `flux_rmse`, `flux_ratio`, `amin_m2`, `best_aic`, `best_bic` |
| `submitted_at` | TIMESTAMP WITH TIME ZONE | NOT NULL, DEFAULT now() | When the update was received by the server |

**Indexes:**
- `PRIMARY KEY (id)`
- Recommended (not yet defined): `INDEX (round_id)`, `INDEX (site_id, round_id)`

**`local_metrics` JSON structure:**
```json
{
  "flux_rmse": 1.23,
  "flux_ratio": 0.71,
  "amin_m2": 0.0042,
  "best_aic": -123.4,
  "best_bic": -119.8
}
```

**Note:** Raw gradients (`delta_W`) are NOT persisted. They are held in `RoundManager._updates` in-memory during the collection phase and discarded after aggregation. This is intentional — gradient tensors can be large and storing them provides no regulatory benefit.

**SQLAlchemy definition:**
```python
class ModelUpdateRecord(Base):
    __tablename__ = "model_updates"
    id:            Mapped[int]      = mapped_column(Integer, primary_key=True)
    site_id:       Mapped[str]      = mapped_column(String(50), nullable=False)
    round_id:      Mapped[int]      = mapped_column(Integer, nullable=False)
    n_samples:     Mapped[int]      = mapped_column(Integer, nullable=False)
    hermia_model:  Mapped[str]      = mapped_column(String(30), default="combined_1a")
    local_metrics: Mapped[dict]     = mapped_column(JSON, default=dict)
    submitted_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
```

---

## Table: `revoked_tokens`

Persisted refresh-token revocation list. One row per consumed or explicitly revoked JWT.  
Used to prevent refresh-token replay attacks.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTO | Surrogate key |
| `jti` | VARCHAR(64) | UNIQUE, NOT NULL | JWT ID claim (UUID4 hex, 32 chars) |
| `site_id` | VARCHAR(50) | NOT NULL | Site that owned the token; for auditing |
| `expires_at` | TIMESTAMP WITH TIME ZONE | NOT NULL | Token's original expiry (from `exp` claim); used for housekeeping |
| `revoked_at` | TIMESTAMP WITH TIME ZONE | NOT NULL, DEFAULT now() | When the token was revoked or consumed |

**Indexes:**
- `PRIMARY KEY (id)`
- `UNIQUE (jti)` — `ix_revoked_tokens_jti` — used for O(1) revocation check on every `/auth/refresh` call

**Token lifecycle:**

```
Issue refresh token (jti=X)
  │
  ├── POST /auth/refresh  →  SELECT revoked WHERE jti=X
  │                          (not found → valid)
  │                          INSERT revoked (jti=X, ...)
  │                          Issue new AT + RT (jti=Y)
  │
  └── POST /auth/revoke   →  SELECT revoked WHERE jti=X
                             (not found → INSERT revoked)
                             Response: {"status": "revoked"}
```

**Housekeeping:** Rows where `expires_at < now()` are safe to delete — the underlying JWT would fail signature verification regardless. A periodic cleanup job (not yet implemented) should prune these rows.

**SQLAlchemy definition:**
```python
class RevokedToken(Base):
    __tablename__ = "revoked_tokens"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True)
    jti:        Mapped[str]      = mapped_column(String(64), unique=True, nullable=False)
    site_id:    Mapped[str]      = mapped_column(String(50), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_revoked_tokens_jti", "jti"),)
```

---

## Initialisation

The database is initialised by `scripts/init_db.py`:

1. `Base.metadata.create_all(engine)` — creates all tables if they do not exist
2. For each site (`site_1` … `site_5`): read `SITE_N_SECRET` from environment, hash with bcrypt, `INSERT OR IGNORE INTO site_registry`
3. Logs generated secrets to stdout (if auto-generated) for the operator to copy to client `.env` files

Subsequent runs are idempotent — existing rows are not overwritten.

In Docker, `init_db.py` is called automatically in the server container's CMD before uvicorn starts.

---

## Migration Workflow

```bash
# After changing server/db/models.py:
alembic revision --autogenerate -m "add last_seen to site_registry"
alembic upgrade head

# Check current migration state:
alembic current

# Downgrade one step:
alembic downgrade -1
```

Migration scripts live in `server/db/migrations/versions/`.  
Never edit a migration that has already been applied to a production database.

---

## PostgreSQL Production Setup

The Docker Compose DB service creates the database automatically:

```yaml
environment:
  POSTGRES_USER:     viral_fl
  POSTGRES_PASSWORD: viral_fl_pass  # change in production
  POSTGRES_DB:       viral_fl
```

Connection string (set in server `.env`):
```
SERVER_DB_URL=postgresql+asyncpg://viral_fl:viral_fl_pass@db:5432/viral_fl
```

For production, use a managed PostgreSQL service (AWS RDS, Azure Database for PostgreSQL, etc.) and set `SERVER_DB_URL` to the managed endpoint. Ensure the password is changed from the default.
