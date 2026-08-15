"""Thread-safe shared training state — written by scheduler, read by UI polling loop."""
from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingState:
    current_round_id: int = 0
    phase: str = "idle"          # idle | training | uploading | done | error
    last_lrv: Optional[float] = None
    last_amin: Optional[float] = None
    last_flux_ratio: Optional[float] = None
    last_hermia_model: Optional[str] = None
    last_round_completed: int = 0
    run_count:            int            = 0
    last_run_at:          Optional[str]  = None  # ISO-8601 UTC string, e.g. "2026-08-14T14:32:00+00:00"


_lock = threading.Lock()
_state = TrainingState()


def get_state() -> TrainingState:
    """Return a snapshot copy of the current training state (thread-safe)."""
    with _lock:
        return dataclasses.replace(_state)


def update_state(**kwargs: object) -> None:
    """Update one or more fields in the shared training state (thread-safe)."""
    valid_fields = {f.name for f in dataclasses.fields(_state)}
    with _lock:
        for key, value in kwargs.items():
            if key not in valid_fields:
                raise AttributeError(f"TrainingState has no field {key!r}")
            setattr(_state, key, value)
