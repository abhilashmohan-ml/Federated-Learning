# client/engine/scheduler.py
"""FL round watcher — dev mode: server-initiated rounds; prod mode: data-directory polling."""
from __future__ import annotations

import time
import threading
from datetime import datetime, timezone

from client.comms.fl_client       import FLClient
from client.engine.data_source    import DataSource, DevDataSource, ProdDataSource, NoNewDataError
from client.engine.local_trainer  import LocalTrainer
from client.engine.state          import get_state, update_state
from shared.utils.logging_config  import get_logger

log = get_logger(__name__)

POLL_SECONDS = 15   # dev mode: how often to check server for new round


def _watch_dev(fl: FLClient, trainer: LocalTrainer) -> None:
    """Dev mode: poll server for server-initiated rounds, train with fresh simulated data."""
    last_seen_round = 0

    while True:
        try:
            data = fl.get_round_status(last_seen_round + 1)
            if data is not None:
                rid    = data.get("round_id", 0)
                status = data.get("status", "")

                if rid > last_seen_round and status == "collecting":
                    log.info("new_round_dev", round_id=rid)
                    update_state(phase="training", current_round_id=rid)
                    update = trainer.train_and_prepare_update(rid)

                    update_state(phase="uploading")
                    fl.upload_update(update)

                    now = datetime.now(timezone.utc).isoformat()
                    state = get_state()
                    update_state(
                        phase="done",
                        last_round_completed=rid,
                        run_count=state.run_count + 1,
                        last_run_at=now,
                        last_lrv=update.local_metrics.get("lrv"),
                        last_flux_ratio=update.local_metrics.get("flux_ratio"),
                        last_amin=update.local_metrics.get("amin_m2"),
                        last_hermia_model=update.hermia_best_model,
                    )
                    last_seen_round = rid

        except Exception as exc:
            update_state(phase="error")
            log.warning("scheduler_poll_error", error=str(exc))

        time.sleep(POLL_SECONDS)


def _watch_prod(
    fl: FLClient,
    trainer: LocalTrainer,
    prod_source: ProdDataSource,
    poll_seconds: int,
) -> None:
    """Prod mode: poll data directory; push update to server when new CSVs arrive."""
    while True:
        try:
            if prod_source.has_new_data():
                update_state(phase="training")
                current_round = fl.get_current_round()
                update = trainer.train_and_prepare_update(current_round.round_id)

                update_state(phase="uploading")
                fl.upload_update(update)

                now = datetime.now(timezone.utc).isoformat()
                state = get_state()
                update_state(
                    phase="done",
                    last_round_completed=current_round.round_id,
                    run_count=state.run_count + 1,
                    last_run_at=now,
                    last_lrv=update.local_metrics.get("lrv"),
                    last_flux_ratio=update.local_metrics.get("flux_ratio"),
                    last_amin=update.local_metrics.get("amin_m2"),
                    last_hermia_model=update.hermia_best_model,
                )
        except NoNewDataError:
            pass    # expected — no data this cycle
        except Exception as exc:
            update_state(phase="error")
            log.warning("prod_poll_error", error=str(exc))

        time.sleep(poll_seconds)


def start_scheduler(data_source: DataSource, fl_client: FLClient) -> None:
    """
    Start the appropriate scheduler thread based on data source type.

    Dev mode (DevDataSource)  → _watch_dev thread, only when auto_schedule=True.
                                When auto_schedule=False (default), training is
                                triggered exclusively via the UI button per site.
    Prod mode (ProdDataSource) → _watch_prod thread always (data-driven).
    """
    from client.config import get_client_settings
    settings = get_client_settings()
    fl       = fl_client
    trainer  = LocalTrainer(data_source=data_source)

    if isinstance(data_source, ProdDataSource):
        target = lambda: _watch_prod(fl, trainer, data_source, settings.data_poll_seconds)
        name   = "fl-scheduler-prod"
    else:
        if not settings.auto_schedule:
            log.info("scheduler_skipped", reason="manual_mode_dev")
            return
        target = lambda: _watch_dev(fl, trainer)
        name   = "fl-scheduler-dev"

    threading.Thread(target=target, daemon=True, name=name).start()
    log.info("scheduler_started", mode="prod" if isinstance(data_source, ProdDataSource) else "dev")
