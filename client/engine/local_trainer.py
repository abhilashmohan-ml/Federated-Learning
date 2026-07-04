"""
Local training engine for one manufacturing site.

Steps
-----
1. Load local flux/pressure/LRV data  (stays on-site)
2. Fit all Hermia blocking models; select best by AIC
3. Compute flux ratio and Amin
4. Apply Gaussian DP noise to fitted parameters
5. Return ModelUpdate payload  (no raw data included)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from client.config          import get_client_settings
from client.engine.data_loader import load_filtration_csv
from shared.models.hermia   import fit_all_models, compute_flux_ratio, compute_amin
from shared.crypto.noise    import add_gaussian_noise
from shared.schemas.federation import ModelUpdate
from shared.utils.logging_config import get_logger

log = get_logger(__name__)


class LocalTrainer:
    def __init__(self) -> None:
        self.settings = get_client_settings()

    def train_and_prepare_update(self, round_id: int) -> ModelUpdate:
        """Run local fitting and return an update safe to send to the server."""
        time, flux, tmp = load_filtration_csv(self.settings.local_data_path)

        # ── Hermia model selection ─────────────────────────────────────────
        results = fit_all_models(time, flux)
        best = next((r for r in results.values() if r.selected), list(results.values())[0])

        # ── Derived metrics ────────────────────────────────────────────────
        flux_ratio = compute_flux_ratio(flux)
        avg_flux   = float(np.mean(flux))
        amin       = compute_amin(
            target_throughput_L=10.0,
            avg_flux_lmh=avg_flux,
            operation_time_h=float(time[-1]) / 60.0,
        )

        local_metrics: Dict[str, float] = {
            "flux_rmse":  best.rmse,
            "flux_ratio": flux_ratio,
            "amin_m2":    amin,
            "best_aic":   best.aic,
            "best_bic":   best.bic,
        }

        # ── Build delta_W from fitted params + apply DP noise ─────────────
        delta_W: Dict[str, List[float]] = {
            "hermia_params": list(best.params.values()),
        }
        delta_W = add_gaussian_noise(delta_W, sigma=self.settings.dp_noise_sigma)

        log.info(
            "local_training_complete",
            site=self.settings.site_id,
            round_id=round_id,
            best_model=best.model_name,
            flux_ratio=f"{flux_ratio:.3f}",
            amin=f"{amin:.4f}",
        )

        return ModelUpdate(
            site_id=self.settings.site_id,
            round_id=round_id,
            n_samples=len(flux),
            delta_W=delta_W,
            dp_noise_sigma=self.settings.dp_noise_sigma,
            hermia_best_model=best.model_name,
            local_metrics=local_metrics,
        )
