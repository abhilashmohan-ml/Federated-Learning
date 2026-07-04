"""
End-to-end local FL simulation  (no network, no Docker required).

Simulates 3 federation rounds across all 5 sites using their
synthetic CSV data, printing aggregation results to console.

Run from project root (after generate_synthetic_data.py):
    python scripts/run_simulation.py
"""
import os
import numpy as np

# Provide minimal env so configs don't fail
os.environ.setdefault("SITE_ID",          "sim")
os.environ.setdefault("SERVER_URL",       "http://localhost:8000")
os.environ.setdefault("SITE_SECRET",      "secret_site_1")
os.environ.setdefault("LOCAL_DATA_PATH",  "data/site_1/filtration.csv")

from shared.models.hermia       import fit_all_models, compute_flux_ratio, compute_amin
from shared.crypto.noise        import add_gaussian_noise
from shared.utils.logging_config import configure_logging, get_logger
from client.engine.data_loader  import load_filtration_csv

configure_logging()
log = get_logger("simulation")

FL_ROUNDS = 3
SITES     = [f"site_{i}" for i in range(1, 6)]


def run() -> None:
    global_W: dict = {}

    for rnd in range(1, FL_ROUNDS + 1):
        log.info("round_start", round=rnd)
        updates = []

        for site in SITES:
            path = f"data/{site}/filtration.csv"
            try:
                time, flux, tmp = load_filtration_csv(path)
            except FileNotFoundError:
                log.warning("data_missing", site=site)
                continue

            results    = fit_all_models(time, flux)
            best       = next((r for r in results.values() if r.selected), list(results.values())[0])
            flux_ratio = compute_flux_ratio(flux)
            amin       = compute_amin(10.0, float(np.mean(flux)), float(time[-1]) / 60.0)
            delta_W    = add_gaussian_noise(
                {"hermia_params": list(best.params.values())}, sigma=0.01
            )
            updates.append({
                "site":       site,
                "n_samples":  len(flux),
                "model":      best.model_name,
                "rmse":       best.rmse,
                "flux_ratio": flux_ratio,
                "amin_m2":    amin,
                "delta_W":    delta_W,
            })
            log.info("site_done", site=site, model=best.model_name,
                     rmse=f"{best.rmse:.3f}", flux_ratio=f"{flux_ratio:.3f}")

        # FedAvg aggregation
        if updates:
            total = sum(u["n_samples"] for u in updates)
            all_params = np.zeros(len(updates[0]["delta_W"]["hermia_params"]))
            for u in updates:
                w = u["n_samples"] / total
                all_params += w * np.array(u["delta_W"]["hermia_params"])
            global_W["hermia_params"] = all_params.tolist()
            log.info("round_complete", round=rnd, n_sites=len(updates),
                     global_params=[f"{v:.4f}" for v in all_params])

    log.info("simulation_finished", final_global_W=global_W)


if __name__ == "__main__":
    run()
