"""
Initialise the server database and register the 5 default sites.

Run once before starting the server:
    python scripts/init_db.py

Site secrets are read from env vars SITE_1_SECRET … SITE_5_SECRET.
If a var is absent a random 32-byte secret is generated and printed
once — copy it to the matching client's SITE_SECRET env var.
"""
import asyncio
import os
import secrets

from passlib.context import CryptContext

from shared.utils.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger("init_db")
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def main() -> None:
    from server.db.database import Base, engine
    from server.db.models import SiteRegistry
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select

    log.info("creating_tables")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    log.info("registering_sites")
    async with AsyncSession(engine) as session:
        for i in range(1, 6):
            sid = f"site_{i}"
            env_var = f"SITE_{i}_SECRET"
            plaintext = os.environ.get(env_var)
            if plaintext is None:
                plaintext = secrets.token_hex(32)
                log.warning(
                    "secret_generated",
                    site=sid,
                    env_var=env_var,
                    secret=plaintext,
                    note="Set this as SITE_SECRET in the client env",
                )
            existing = await session.execute(
                select(SiteRegistry).where(SiteRegistry.site_id == sid)
            )
            if existing.scalar_one_or_none() is None:
                session.add(SiteRegistry(
                    site_id=sid,
                    secret_hash=_pwd.hash(plaintext),
                ))
            else:
                log.info("site_already_registered", site=sid)
        await session.commit()

    log.info("init_complete")


if __name__ == "__main__":
    asyncio.run(main())
