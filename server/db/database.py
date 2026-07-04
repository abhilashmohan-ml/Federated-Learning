"""
SQLAlchemy async database engine and session factory.

WHY ASYNC DATABASE ACCESS?
---------------------------
FastAPI is an async web framework — it can handle many concurrent requests
without blocking. If we used a synchronous database driver (the traditional
kind), every database query would block the entire server while waiting for
the DB to respond. With an async driver (asyncpg for PostgreSQL, aiosqlite
for SQLite), FastAPI can serve other requests while waiting for the DB.

PYTHON CONCEPT: async/await
  Python's `async` keyword marks a function as a "coroutine" — a function
  that can pause and resume, allowing other code to run while it waits
  (for example, for a database query to return).
  `await some_coroutine()` means "pause this function here and wait for
  some_coroutine to finish, but let other code run in the meantime."

PYTHON CONCEPT: Generator / yield in async context
  `get_db()` uses `yield` instead of `return`. This makes it an async
  generator — useful for dependency injection where you want to set something
  up, hand it to the caller, and then tear it down afterwards.

  How FastAPI uses get_db():
    1. FastAPI calls `get_db()`
    2. The function creates a session with `async with AsyncSessionLocal()`
    3. It `yield`s the session to the endpoint function
    4. The endpoint uses the session for its queries
    5. When the endpoint returns, the `async with` block exits, which
       automatically commits pending changes and closes the session

HOW THE CONNECTION POOL WORKS
--------------------------------
`create_async_engine` creates a pool of database connections. Rather than
opening and closing a new connection per request (slow), the engine reuses
idle connections from the pool. The default pool size is usually 5-10.

`AsyncSessionLocal` is a factory that creates new session objects from
the engine. Sessions are not the same as connections — one session can
use multiple connections over its lifetime, and the engine manages them.

`expire_on_commit=False` prevents SQLAlchemy from immediately expiring
(invalidating) objects after a commit. In async code, accessing an expired
object would trigger a new DB query outside the session context, causing
errors. With this option, attributes remain accessible after commit.
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,          # type hint for async DB sessions
    async_sessionmaker,    # factory that creates new sessions
    create_async_engine,   # connects to the database
)
from sqlalchemy.orm import DeclarativeBase   # base class for all ORM models

from server.config import get_settings

settings = get_settings()

# ── Engine ─────────────────────────────────────────────────────────────────────
#
# The engine is the core of SQLAlchemy's connection pooling. It holds a pool
# of live database connections and dispatches them to sessions on demand.
#
# `echo=False` — set to True temporarily to see every SQL query in the logs.
#   Useful for debugging but very noisy in production.
#
# db_url examples:
#   sqlite+aiosqlite:///./viral_fl.db     → SQLite file in the current directory
#   postgresql+asyncpg://user:pw@db:5432/viral_fl → PostgreSQL (production)
engine = create_async_engine(settings.db_url, echo=False)

# ── Session factory ────────────────────────────────────────────────────────────
#
# `async_sessionmaker` creates a "template" for making new sessions.
# Every call to `AsyncSessionLocal()` produces a new session bound to
# the engine above.
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# ── ORM base class ─────────────────────────────────────────────────────────────
#
# All ORM model classes in server/db/models.py inherit from this Base.
# SQLAlchemy uses the Base to keep track of all defined tables and generate
# the SQL for `CREATE TABLE`, `DROP TABLE`, and Alembic migrations.
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models in this project."""
    pass


# ── Session dependency for FastAPI ────────────────────────────────────────────
#
# FastAPI's dependency injection system will call this function for any
# endpoint that declares `db: AsyncSession = Depends(get_db)`.
# The function sets up a session, hands it to the endpoint via `yield`,
# then closes it when the endpoint finishes.
async def get_db() -> AsyncSession:  # type: ignore[misc]
    """
    Async generator that provides a database session per request.

    Usage in a FastAPI endpoint:
        @router.get("/example")
        async def my_endpoint(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(SiteRegistry))
            return result.scalars().all()

    The `async with` block guarantees the session is always closed,
    even if an exception is raised inside the endpoint.
    """
    async with AsyncSessionLocal() as session:
        yield session   # hand the session to the endpoint; resume here on exit
