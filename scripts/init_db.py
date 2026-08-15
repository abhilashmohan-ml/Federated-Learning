"""
Initialise the server database and register manufacturing sites.

Run once before starting the server:
    python scripts/init_db.py

Sites are read from the REGISTERED_SITES env var (comma-separated
site_id:secret pairs).  If the secret is omitted a random 32-byte secret
is generated and printed — copy it to that site's SITE_SECRET env var.

Example:
    REGISTERED_SITES=basel:s3cr3t,singapore:another_secret python scripts/init_db.py
    REGISTERED_SITES=site_1:,site_2:,site_3:  # generate secrets for all three
"""
import asyncio
import os
import secrets

import bcrypt

from shared.utils.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger("init_db")


def _hash(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt()).decode()


def _parse_registered_sites(raw: str) -> list[tuple[str, str]]:
    """Parse 'site_a:secret_a,site_b:secret_b' into [(site_id, secret), ...]."""
    result = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            site_id, _, secret = entry.partition(":")
        else:
            site_id, secret = entry, ""
        site_id = site_id.strip()
        secret = secret.strip()
        if site_id:
            result.append((site_id, secret))
    return result


async def main() -> None:
    from server.db.database import Base, engine
    from server.db.models import SiteRegistry
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select

    registered_raw = os.environ.get("REGISTERED_SITES", "")
    if not registered_raw.strip():
        log.error(
            "REGISTERED_SITES_missing",
            note="Set REGISTERED_SITES=site_a:secret_a,site_b:secret_b and re-run",
        )
        raise SystemExit(1)

    sites = _parse_registered_sites(registered_raw)
    if not sites:
        log.error("REGISTERED_SITES_empty", raw=registered_raw)
        raise SystemExit(1)

    log.info("creating_tables")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    log.info("registering_sites", count=len(sites))
    async with AsyncSession(engine) as session:
        for sid, plaintext in sites:
            if not plaintext:
                plaintext = secrets.token_hex(32)
                log.warning(
                    "secret_generated",
                    site=sid,
                    secret=plaintext,
                    note="Set this as SITE_SECRET in the client env",
                )
            existing = await session.execute(
                select(SiteRegistry).where(SiteRegistry.site_id == sid)
            )
            if existing.scalar_one_or_none() is None:
                session.add(SiteRegistry(
                    site_id=sid,
                    secret_hash=_hash(plaintext),
                ))
            else:
                log.info("site_already_registered", site=sid)
        await session.commit()

    log.info("init_complete")


if __name__ == "__main__":
    asyncio.run(main())
