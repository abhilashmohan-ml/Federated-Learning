"""SQLAlchemy ORM models.  Migrate with Alembic — never alter manually."""
from datetime import datetime
from sqlalchemy import DateTime, Float, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column
from server.db.database import Base


class SiteRegistry(Base):
    __tablename__ = "site_registry"
    id: Mapped[int]            = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str]       = mapped_column(String(50), unique=True, nullable=False)
    secret_hash: Mapped[str]   = mapped_column(String(256), nullable=False)
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RoundRecord(Base):
    __tablename__ = "rounds"
    id: Mapped[int]            = mapped_column(Integer, primary_key=True)
    round_id: Mapped[int]      = mapped_column(Integer, unique=True, nullable=False)
    status: Mapped[str]        = mapped_column(String(20), default="pending")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    model_version: Mapped[int] = mapped_column(Integer, default=0)
    global_metrics: Mapped[dict] = mapped_column(JSON, default=dict)


class ModelUpdateRecord(Base):
    __tablename__ = "model_updates"
    id: Mapped[int]            = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str]       = mapped_column(String(50), nullable=False)
    round_id: Mapped[int]      = mapped_column(Integer, nullable=False)
    n_samples: Mapped[int]     = mapped_column(Integer, nullable=False)
    hermia_model: Mapped[str]  = mapped_column(String(30), default="combined_1a")
    local_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
