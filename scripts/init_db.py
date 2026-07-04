"""
Initialise the server database and register the 5 default sites.

Run once before starting the server:
    python scripts/init_db.py
"""
import asyncio
from shared.utils.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger("init_db")


async def main() -> None:
    from server.db.database import engine, Base
    from server.db.models   import SiteRegistry
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select

    log.info("creating_tables")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    log.info("registering_sites")
    async with AsyncSession(engine) as session:
        for i in range(1, 6):
            sid = f"site_{i}"
            existing = await session.execute(
                select(SiteRegistry).where(SiteRegistry.site_id == sid)
            )
            if existing.scalar_one_or_none() is None:
                session.add(SiteRegistry(
                    site_id=sid,
                    secret_hash=f"secret_site_{i}",   # hash in production
                ))
        await session.commit()

    log.info("init_complete")


if __name__ == "__main__":
    asyncio.run(main())
