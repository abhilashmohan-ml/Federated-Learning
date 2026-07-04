"""
Physics-Informed Neural Network (PINN) for viral filtration.

Architecture
------------
Level 1 — Parameter Predictor Network
    Input : feature vector x  (filter descriptors + process conditions)
    Output: physically meaningful parameters
            {J0, ks, ki, kc, kcf, k1, k2, Pc, Jcrit, Dv}

Level 2 — Physics Solver (differentiable)
    Uses Level 1 outputs in Hermia / Manabe / polarization equations
    to produce J(t), LRV, Amin predictions.

Loss
----
    L_total = L_flux + L_LRV + L_physics + L_fedprox

    L_flux    = MSE(J_pred(t), J_obs(t))
    L_LRV     = MSE(LRV_pred, LRV_obs)
    L_physics = constraint penalties (J0>0, 0<=Pc<=1, k>0, ...)
    L_fedprox = (mu/2) * ||W_local - W_global||^2
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from shared.utils.constants import N_PARAMS, PARAM_IDX


# ── Feature dimensions ────────────────────────────────────────────────────────
#  Filter descriptors  : pore_size_nm, nmwco_kda, membrane_area_m2  (3)
#  Process conditions  : tmp_bar, feed_flux_lmh, pH, IS_mM,
#                        mab_conc_g_L, temperature_C              (6)
#  Virus properties    : virus_size_nm, virus_charge               (2)
#  Total input dim     : 11
INPUT_DIM = 11


class ParameterPredictor(nn.Module):
    """Maps process/filter features to mechanistic parameters."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, N_PARAMS),
        )
        # Positive-constraint activations applied in forward()
        self._softplus = nn.Softplus()
        self._sigmoid  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)                          # (B, N_PARAMS)
        params = torch.zeros_like(raw)
        # All k-values and J0 must be positive
        for key in ("J0", "ks", "ki", "kc", "kcf", "k1", "k2", "Jcrit", "Dv"):
            i = PARAM_IDX[key]
            params[:, i] = self._softplus(raw[:, i]) + 1e-6
        # Pc must be in (0, 1)
        params[:, PARAM_IDX["Pc"]] = self._sigmoid(raw[:, PARAM_IDX["Pc"]])
        return params


class BlockingRegimeClassifier(nn.Module):
    """Classifies the dominant Hermia blocking regime (5 classes)."""

    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),   # standard / complete / intermediate / cake / combined_1a
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)      # raw logits; use CrossEntropyLoss


class PhysicsSolver(nn.Module):
    """Differentiable physics layer — no learnable weights."""

    def forward(
        self,
        params: torch.Tensor,   # (B, N_PARAMS)
        time: torch.Tensor,     # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        J_t  : flux decay  (B, T)
        LRV  : log reduction value  (B,)
        """
        J0    = params[:, PARAM_IDX["J0"]].unsqueeze(1)    # (B, 1)
        k1    = params[:, PARAM_IDX["k1"]].unsqueeze(1)
        k2    = params[:, PARAM_IDX["k2"]].unsqueeze(1)
        Pc    = params[:, PARAM_IDX["Pc"]]                  # (B,)

        # Combined 1-A flux
        J_t = (J0 / (1.0 + k1 * time) ** 2) * torch.exp(-k2 * time)  # (B, T)

        # Manabe LRV  (single layer)
        Pc_c = torch.clamp(Pc, 1e-7, 1.0 - 1e-7)
        LRV  = torch.log10(1.0 / (1.0 - Pc_c))             # (B,)

        return J_t, LRV


class FiltrationPINN(nn.Module):
    """Full Physics-Informed Neural Network for viral filtration."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.predictor  = ParameterPredictor(input_dim, hidden)
        self.classifier = BlockingRegimeClassifier(input_dim)
        self.solver     = PhysicsSolver()

    def forward(
        self,
        x: torch.Tensor,        # (B, input_dim)
        time: torch.Tensor,     # (B, T) — time points [minutes]
    ) -> dict[str, torch.Tensor]:
        params         = self.predictor(x)
        regime_logits  = self.classifier(x)
        J_t, LRV       = self.solver(params, time)
        return {
            "J_t":           J_t,
            "LRV":           LRV,
            "params":        params,
            "regime_logits": regime_logits,
        }


# ── Loss function ─────────────────────────────────────────────────────────────

def filtration_loss(
    outputs: dict[str, torch.Tensor],
    J_obs: torch.Tensor,
    LRV_obs: torch.Tensor,
    regime_labels: Optional[torch.Tensor],
    global_weights: Optional[dict[str, torch.Tensor]],
    local_weights:  Optional[dict[str, torch.Tensor]],
    fedprox_mu: float = 0.01,
    lambda_physics: float = 1.0,
) -> torch.Tensor:
    """
    L_total = L_flux + L_LRV + L_physics + L_fedprox
    """
    # Flux MSE
    L_flux = nn.functional.mse_loss(outputs["J_t"], J_obs)

    # LRV MSE
    L_LRV = nn.functional.mse_loss(outputs["LRV"], LRV_obs)

    # Physics constraint penalties  (soft — penalise violations)
    params = outputs["params"]
    L_physics = (
        torch.relu(-params).sum()                                      # all params >= 0
        + torch.relu(params[:, PARAM_IDX["Pc"]] - 1.0).sum()          # Pc <= 1
    ) * lambda_physics

    # Blocking regime classification loss (optional)
    L_regime = torch.tensor(0.0)
    if regime_labels is not None:
        L_regime = nn.functional.cross_entropy(outputs["regime_logits"], regime_labels)

    # FedProx proximal term
    L_fedprox = torch.tensor(0.0)
    if global_weights is not None and local_weights is not None:
        prox = sum(
            torch.sum((local_weights[k] - global_weights[k]) ** 2)
            for k in global_weights
            if k in local_weights
        )
        L_fedprox = (fedprox_mu / 2.0) * prox

    return L_flux + L_LRV + L_physics + L_regime + L_fedprox
