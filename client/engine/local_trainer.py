"""
Local training engine — runs on each manufacturing site.

WHAT HAPPENS IN LOCAL TRAINING?
---------------------------------
Each FL round, the local trainer performs the following steps on the site's
private data WITHOUT sending the raw data anywhere:

  1. Load local filtration data (time, flux, TMP) from the CSV file
  2. Fit ALL 5 Hermia blocking models to the measured flux decline
  3. Use AIC (Akaike Information Criterion) to select the best model
  4. Compute derived process metrics (flux ratio, minimum filter area)
  5. Package the fitted parameters as a weight DELTA (delta_W)
  6. Add Gaussian differential privacy (DP) noise to the delta
  7. Return a ModelUpdate containing the noisy delta + metadata

The key privacy property: raw measurements (time, flux, TMP) are used only
within this function and are NEVER included in the returned ModelUpdate.
Only the model parameters (which are aggregate statistics, not individual
measurements) are shared — and even those are protected by DP noise.

WHAT IS delta_W?
-----------------
In standard neural network training, we share the full model weights W.
In federated learning, we share the CHANGE in weights: delta_W = W_local - W_global.
This is algebraically equivalent but:
  - Communicates less information per round
  - Allows the aggregation to be expressed as a weighted average of changes
    rather than a weighted average of full weights

Currently, `delta_W` holds the Hermia model parameters rather than true neural
network weight deltas. A future version will integrate the PINN's actual weights.

DIFFERENTIAL PRIVACY
---------------------
We add zero-mean Gaussian noise to delta_W before sending it to the server.
This provides plausible deniability — even if an adversary intercepts multiple
rounds of updates, they cannot reconstruct the exact local measurements because
the noise drowns out the fine-grained details.

The sigma parameter (DP_NOISE_SIGMA env var) controls the trade-off:
  - sigma=0.001 → very small noise, high accuracy, weaker privacy
  - sigma=0.1   → large noise, lower accuracy, stronger privacy
  - sigma=0.01  → balanced default for pharmaceutical manufacturing

PYTHON CONCEPT: generator expression with next()
  `next((r for r in results.values() if r.selected), ...)` steps through the
  HermiaResult objects lazily and returns the first one where `.selected` is
  True. The second argument to next() is a fallback value if none is selected.
  This is more concise than a for loop with a break.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from client.config              import get_client_settings
from client.engine.data_loader  import load_filtration_csv
from shared.models.hermia       import fit_all_models, compute_flux_ratio, compute_amin
from shared.crypto.noise        import add_gaussian_noise
from shared.schemas.federation  import ModelUpdate
from shared.utils.logging_config import get_logger

log = get_logger(__name__)


class LocalTrainer:
    """
    Orchestrates local model fitting and delta construction for one FL round.

    Each call to `train_and_prepare_update()` is a self-contained training
    episode: load data → fit models → apply DP noise → return update.

    The class holds no mutable state — all state is in the settings object
    (read-only) and the local variables of each method call.
    """

    def __init__(self) -> None:
        # Get the (cached) settings object — same object shared across all modules
        self.settings = get_client_settings()

    def train_and_prepare_update(self, round_id: int) -> ModelUpdate:
        """
        Execute one local training cycle and return a privacy-protected update.

        This is the main method called by the scheduler for each FL round.
        It takes the round_id as input (to label the update) and returns
        a ModelUpdate object ready to POST to the server.

        Parameters
        ----------
        round_id : int — the ID of the FL round we are training for.
                         Included in the update so the server can match it
                         to the correct round.

        Returns
        -------
        ModelUpdate — contains:
            site_id      : which site this is
            round_id     : which round this is for
            n_samples    : number of data points (used as aggregation weight)
            delta_W      : noisy model parameter changes (SAFE TO TRANSMIT)
            hermia_best_model : which Hermia model won the AIC competition
            local_metrics: flux RMSE, flux ratio, Amin (for server dashboard)
            dp_noise_sigma: sigma used for DP noise (for audit log)
        """
        # ── Step 1: Load local data (stays on-site) ────────────────────────────
        time, flux, tmp = load_filtration_csv(self.settings.local_data_path)

        # ── Step 2: Fit all 5 Hermia models ────────────────────────────────────
        # Each model is fit to the (time, flux) data independently.
        # Returns a dict of model_name → HermiaResult with AIC, BIC, RMSE, params.
        results = fit_all_models(time, flux)

        # ── Step 3: Select best model by AIC ────────────────────────────────────
        # `r.selected` is True for exactly one model (the one with the lowest AIC).
        # If somehow no model was selected (e.g., all fitters failed), fall back
        # to the first available result.
        best = next(
            (r for r in results.values() if r.selected),  # first selected model
            list(results.values())[0],                     # fallback: first model
        )

        # ── Step 4: Compute derived process metrics ─────────────────────────────
        flux_ratio = compute_flux_ratio(flux)
        avg_flux   = float(np.mean(flux))   # LMH average over the whole run

        # Minimum filter area for a 10-litre batch given this site's average flux
        amin = compute_amin(
            target_throughput_L=10.0,
            avg_flux_lmh=avg_flux,
            operation_time_h=float(time[-1]) / 60.0,  # convert last time point to hours
        )

        # Collect all metrics we want to report to the server dashboard
        local_metrics: Dict[str, float] = {
            "flux_rmse":  best.rmse,        # how well the best model fit the data
            "flux_ratio": flux_ratio,        # J_final / J_initial (fouling severity)
            "amin_m2":    amin,              # minimum filter area in m²
            "best_aic":   best.aic,          # AIC of the selected model
            "best_bic":   best.bic,          # BIC of the selected model
        }

        # ── Step 5: Build delta_W from fitted model parameters ─────────────────
        # We package the best model's fitted parameter values as the "weight delta."
        # For the Combined 1-A model, this would be {"hermia_params": [J0, k1, k2]}.
        # For Standard blocking: {"hermia_params": [J0, ks]}.
        #
        # NOTE: In a full PINN implementation, delta_W would contain the actual
        # neural network weight changes (W_local - W_global). The current version
        # uses Hermia parameters as a placeholder.
        delta_W: Dict[str, List[float]] = {
            "hermia_params": list(best.params.values()),
        }

        # ── Step 6: Apply differential privacy noise ────────────────────────────
        # add_gaussian_noise modifies delta_W in-place (conceptually), returning
        # a new dict with noise added to each value. After this step, the exact
        # parameter values are protected — an observer cannot reconstruct them.
        delta_W = add_gaussian_noise(delta_W, sigma=self.settings.dp_noise_sigma)

        log.info(
            "local_training_complete",
            site=self.settings.site_id,
            round_id=round_id,
            best_model=best.model_name,
            flux_ratio=f"{flux_ratio:.3f}",
            amin=f"{amin:.4f}",
        )

        # ── Step 7: Build and return the ModelUpdate ────────────────────────────
        return ModelUpdate(
            site_id=self.settings.site_id,
            round_id=round_id,
            n_samples=len(flux),                         # number of data points (for weighting)
            delta_W=delta_W,                             # noisy parameter changes
            dp_noise_sigma=self.settings.dp_noise_sigma, # auditable DP parameter
            hermia_best_model=best.model_name,           # which model won ("combined_1a", etc.)
            local_metrics=local_metrics,                 # dashboard metrics
        )
