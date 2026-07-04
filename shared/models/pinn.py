"""
Physics-Informed Neural Network (PINN) for viral filtration.

WHAT IS A PINN?
---------------
A standard neural network learns patterns purely from data ("black box").
A Physics-Informed Neural Network (PINN) incorporates known physical equations
directly into its structure and/or loss function, combining the flexibility of
neural networks with the interpretability of mechanistic models.

In this system, the PINN has TWO levels:

  LEVEL 1 — Parameter Predictor Network:
    Input:  11-dimensional feature vector (filter + process + virus properties)
    Output: 10 physical parameters {J0, ks, ki, kc, kcf, k1, k2, Pc, Jcrit, Dv}
    These parameters have PHYSICAL MEANING — they are the same parameters
    that appear in the Hermia and Manabe equations.

  LEVEL 2 — Physics Solver:
    Takes the predicted parameters from Level 1 and plugs them into the known
    physical equations (Combined 1-A flux, Manabe LRV) to produce predictions.
    This layer has NO learnable weights — it is pure mathematics.

WHY THIS ARCHITECTURE?
-----------------------
  - The neural network (Level 1) learns how filter/process properties relate
    to physical model parameters across many different sites and experiments.
  - The physics (Level 2) constrains the output to physically meaningful
    predictions — the model cannot predict negative flux or impossible LRV.
  - This generalises better than a pure black-box model, especially with
    limited training data.

FEDERATED LEARNING CONTEXT
---------------------------
Each site trains a local copy of this PINN on its private data. The LEARNED
WEIGHTS of Level 1 (the Parameter Predictor) are what get shared with the server
as model updates. The physical equations in Level 2 are the same at all sites.

PYTHON CONCEPT: nn.Module
  All neural network components inherit from `torch.nn.Module`. This gives
  them a `.parameters()` method, automatic gradient tracking, and the ability
  to move to GPU with `.cuda()`.

PYTHON CONCEPT: torch.Tensor
  A Tensor is PyTorch's equivalent of a NumPy array — a multi-dimensional array
  of numbers that supports automatic differentiation (gradients).
  Shape notation: (B, T) means a 2D tensor with B rows and T columns.
    B = batch size (number of experiments processed simultaneously)
    T = number of time points per experiment

PYTHON CONCEPT: forward() method
  Every nn.Module must implement `forward(self, ...)`. When you call the
  module as if it were a function (e.g. `output = model(input)`), PyTorch
  automatically calls `forward()` and tracks gradients for backpropagation.
"""

from __future__ import annotations

from typing import Optional

import torch                # PyTorch deep learning framework
import torch.nn as nn       # neural network building blocks (Linear, ReLU, etc.)

from shared.utils.constants import N_PARAMS, PARAM_IDX


# ── Feature dimensions ─────────────────────────────────────────────────────────
#
# The PINN takes 11 input features describing the filter and process.
# These are the same features as ProcessConditions + FilterDescriptor + virus props
# in shared/schemas/filtration.py.
#
# Filter descriptors   (3): pore_size_nm, nmwco_kda, membrane_area_m2
# Process conditions   (6): tmp_bar, feed_flux_lmh, pH, IS_mM, mab_conc_g_L, temperature_C
# Virus properties     (2): virus_size_nm, virus_charge
# ──────────────────────────────────────────────────────────────────────────
# Total                (11)
INPUT_DIM = 11


class ParameterPredictor(nn.Module):
    """
    Level-1 network: maps process/filter features to mechanistic parameters.

    ARCHITECTURE
    ------------
    Input  (B × 11)
      → Linear(11→128) → ReLU
      → Linear(128→128) → ReLU
      → Linear(128→64) → ReLU
      → Linear(64→10)
      → Custom activations per parameter (Softplus for positive, Sigmoid for Pc)

    WHY RELU?
    ---------
    ReLU (Rectified Linear Unit) is the most common activation function in deep
    learning. It replaces negative values with zero: ReLU(x) = max(0, x).
    It is cheap to compute and prevents the "vanishing gradient" problem that
    plagued earlier activations like tanh.

    WHY SOFTPLUS FOR PARAMETERS?
    ----------------------------
    Physical parameters like J0, ks, k1, k2 must be strictly positive. Simply
    using ReLU could produce exactly zero (which causes /0 errors in physics).
    Softplus(x) = ln(1 + exp(x)) is always positive and smooth everywhere.
    We add a small 1e-6 to prevent it from being exactly zero.

    WHY SIGMOID FOR Pc?
    -------------------
    The capture probability Pc must be in [0, 1]. Sigmoid(x) = 1/(1+exp(-x))
    maps any real number to (0, 1) — perfect for probabilities.

    Parameters
    ----------
    input_dim : int — number of input features (default 11)
    hidden    : int — number of neurons in each hidden layer (default 128)
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = 128) -> None:
        super().__init__()   # ALWAYS call super().__init__() first in nn.Module

        # nn.Sequential chains layers in order; the output of each layer
        # becomes the input to the next.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),  # input layer: 11 → 128
            nn.ReLU(),                     # non-linear activation
            nn.Linear(hidden, hidden),     # hidden layer: 128 → 128
            nn.ReLU(),
            nn.Linear(hidden, 64),         # narrowing: 128 → 64
            nn.ReLU(),
            nn.Linear(64, N_PARAMS),       # output layer: 64 → 10 parameters
        )

        # Custom activation functions applied after the linear output
        self._softplus = nn.Softplus()   # smooth positive: ln(1 + exp(x)) > 0
        self._sigmoid  = nn.Sigmoid()    # squashes to (0, 1) for probabilities

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the input features through the network and apply physical constraints.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 11)
            Batch of B feature vectors, each with 11 elements.

        Returns
        -------
        torch.Tensor, shape (B, 10)
            Batch of B parameter vectors, each with 10 physically constrained values.
        """
        raw = self.net(x)   # (B, N_PARAMS) — raw unconstrained outputs

        # We want different constraints for different parameters.
        # Initialise an output tensor of the same shape and device as `raw`.
        params = torch.zeros_like(raw)

        # Apply Softplus + epsilon to all parameters that must be positive
        for key in ("J0", "ks", "ki", "kc", "kcf", "k1", "k2", "Jcrit", "Dv"):
            i = PARAM_IDX[key]   # look up which column this parameter is in
            # Softplus(raw) is always > 0; +1e-6 prevents exact zero
            params[:, i] = self._softplus(raw[:, i]) + 1e-6

        # Apply Sigmoid to Pc so it stays in (0, 1)
        params[:, PARAM_IDX["Pc"]] = self._sigmoid(raw[:, PARAM_IDX["Pc"]])

        return params


class BlockingRegimeClassifier(nn.Module):
    """
    Classifies the dominant Hermia blocking regime from input features.

    This is a standard 5-class classification network. Given the filter and
    process conditions, it predicts WHICH Hermia model best describes the
    fouling behaviour at this site.

    Output classes (0–4):
      0 = standard blocking
      1 = complete blocking
      2 = intermediate blocking
      3 = cake filtration
      4 = combined 1-A

    The output is raw LOGITS (un-normalised scores). To get probabilities,
    apply softmax: probability[i] = exp(logit[i]) / sum(exp(all logits)).
    During training, `nn.CrossEntropyLoss` handles the softmax internally.

    Parameters
    ----------
    input_dim : int — 11 input features (same as ParameterPredictor)
    """

    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  # compress: 11 → 64
            nn.ReLU(),
            nn.Linear(64, 5),          # 5 class logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 11)

        Returns
        -------
        torch.Tensor, shape (B, 5)
            Raw logits for each class. Use cross-entropy loss at training time.
        """
        return self.net(x)


class PhysicsSolver(nn.Module):
    """
    Level-2 physics layer — differentiable physical equations with NO learnable weights.

    This layer takes the parameters predicted by ParameterPredictor and plugs
    them into the physical equations to produce predictions:
      - J(t) : flux at every time point using the Combined 1-A equation
      - LRV  : log reduction value using the Manabe equation

    DIFFERENTIABLE means PyTorch can compute gradients through this layer
    (backpropagation works through the exp() and division operations).
    This is essential so the physics equations contribute to gradient updates.

    WHY torch.clamp?
    ----------------
    Just like np.clip, torch.clamp constrains values to a range.
    Used here to prevent log10(0) which would give -inf gradients.
    """

    def forward(
        self,
        params: torch.Tensor,   # (B, N_PARAMS) — from ParameterPredictor
        time:   torch.Tensor,   # (B, T) — T time points per sample
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply physics equations to predict flux and LRV.

        Parameters
        ----------
        params : torch.Tensor, shape (B, N_PARAMS=10)
            Physical parameters from the ParameterPredictor.
        time   : torch.Tensor, shape (B, T)
            Time points at which to evaluate the flux curve [minutes].

        Returns
        -------
        J_t : torch.Tensor, shape (B, T)
            Predicted flux at each time point for each sample.
        LRV : torch.Tensor, shape (B,)
            Predicted LRV for each sample (single value per sample).
        """
        # Extract specific parameters from the parameter vector.
        # `.unsqueeze(1)` adds a dimension of size 1 at position 1, changing
        # shape from (B,) to (B, 1). This allows broadcasting against (B, T) time.
        J0 = params[:, PARAM_IDX["J0"]].unsqueeze(1)   # (B, 1)
        k1 = params[:, PARAM_IDX["k1"]].unsqueeze(1)   # (B, 1)
        k2 = params[:, PARAM_IDX["k2"]].unsqueeze(1)   # (B, 1)
        Pc = params[:, PARAM_IDX["Pc"]]                 # (B,) — no unsqueeze needed

        # ── Combined 1-A flux equation ─────────────────────────────────────────
        # J(t) = J0 / (1 + k1·t)² × exp(-k2·t)
        # Broadcasting: (B,1) / (B,1 + B,1 × B,T)² × exp(B,1 × B,T) = (B,T)
        J_t = (J0 / (1.0 + k1 * time) ** 2) * torch.exp(-k2 * time)

        # ── Manabe LRV equation ────────────────────────────────────────────────
        # LRV = log₁₀(1 / (1 - Pc))
        # torch.clamp prevents Pc from being exactly 0 or 1 (which would give
        # log10(∞) or log10(0) = undefined/infinite gradients)
        Pc_c = torch.clamp(Pc, 1e-7, 1.0 - 1e-7)
        LRV  = torch.log10(1.0 / (1.0 - Pc_c))   # (B,)

        return J_t, LRV


class FiltrationPINN(nn.Module):
    """
    The complete Physics-Informed Neural Network for viral filtration.

    This is the top-level model that combines all three components:
      1. ParameterPredictor  — maps features to physical parameters
      2. BlockingRegimeClassifier — predicts which fouling mechanism dominates
      3. PhysicsSolver       — turns parameters into flux and LRV predictions

    USAGE IN FEDERATED LEARNING
    ----------------------------
    Each site has its own FiltrationPINN instance. During each FL round:
      1. Site downloads global weights from the server
      2. Local training adjusts the weights on local data for `LOCAL_EPOCHS` epochs
      3. The weight DELTA (change) is packaged with DP noise into a ModelUpdate
      4. The server aggregates all sites' deltas into new global weights

    Parameters
    ----------
    input_dim : int — number of input features (default 11)
    hidden    : int — hidden layer size in ParameterPredictor (default 128)
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.predictor  = ParameterPredictor(input_dim, hidden)
        self.classifier = BlockingRegimeClassifier(input_dim)
        self.solver     = PhysicsSolver()   # no learnable weights

    def forward(
        self,
        x:    torch.Tensor,   # (B, input_dim) — feature vectors
        time: torch.Tensor,   # (B, T)          — time points
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: features → parameters → physics predictions.

        Parameters
        ----------
        x    : torch.Tensor, shape (B, 11)
        time : torch.Tensor, shape (B, T)

        Returns
        -------
        dict with keys:
          "J_t"           : (B, T) predicted flux at each time point
          "LRV"           : (B,) predicted LRV for each sample
          "params"        : (B, 10) predicted physical parameters
          "regime_logits" : (B, 5) blocking regime classification logits
        """
        params        = self.predictor(x)          # Level 1: features → parameters
        regime_logits = self.classifier(x)          # parallel classification head
        J_t, LRV      = self.solver(params, time)   # Level 2: parameters → physics

        return {
            "J_t":           J_t,
            "LRV":           LRV,
            "params":        params,
            "regime_logits": regime_logits,
        }


# ── Loss function ───────────────────────────────────────────────────────────────

def filtration_loss(
    outputs:        dict[str, torch.Tensor],
    J_obs:          torch.Tensor,
    LRV_obs:        torch.Tensor,
    regime_labels:  Optional[torch.Tensor],
    global_weights: Optional[dict[str, torch.Tensor]],
    local_weights:  Optional[dict[str, torch.Tensor]],
    fedprox_mu:     float = 0.01,
    lambda_physics: float = 1.0,
) -> torch.Tensor:
    """
    Compute the total training loss for the PINN.

    The loss has FOUR components:

    L_total = L_flux + L_LRV + L_physics + L_regime + L_fedprox

    1. L_flux    (flux MSE):
       Penalises difference between predicted J(t) and measured J(t).
       MSE = mean((J_pred - J_obs)²)

    2. L_LRV     (LRV MSE):
       Penalises difference between predicted LRV and measured LRV.
       Only computed where LRV measurements are available.

    3. L_physics (constraint penalties):
       "Soft constraints" — penalise parameter values that violate physics:
         - All parameters should be ≥ 0 (relu(-params).sum() penalises negatives)
         - Pc should be ≤ 1 (relu(Pc - 1).sum() penalises Pc > 1)
       If the predictor is well-behaved (Softplus/Sigmoid activations), these
       will be zero — but they act as a safety net.

    4. L_regime  (classification cross-entropy):
       If we know the true fouling mechanism (from Hermia AIC selection), we
       can provide labels and train the classifier. Set to 0 if no labels.

    5. L_fedprox (FedProx proximal term):
       The FedProx algorithm adds a term that prevents local training from
       drifting too far from the global model:
         L_fedprox = (μ/2) × ‖W_local - W_global‖²
       This is the key difference from standard FedAvg — it improves convergence
       when sites have heterogeneous (non-IID) data.

    Parameters
    ----------
    outputs        : dict — outputs from FiltrationPINN.forward()
    J_obs          : torch.Tensor — measured flux (B, T)
    LRV_obs        : torch.Tensor — measured LRV (B,)
    regime_labels  : Optional[torch.Tensor] — true Hermia class indices (B,)
                     None if not available
    global_weights : Optional[dict] — global model weights at start of round
                     None for first round (no prior global model)
    local_weights  : Optional[dict] — current local model weights during training
                     None if FedProx should be skipped
    fedprox_mu     : float — FedProx regularisation strength μ (default 0.01)
                     Higher = stays closer to global model, lower = more local freedom
    lambda_physics : float — physics constraint penalty weight (default 1.0)

    Returns
    -------
    torch.Tensor — scalar loss value to minimise via gradient descent
    """
    # ── 1. Flux reconstruction loss ────────────────────────────────────────────
    # MSE between predicted flux curve and measured flux curve
    L_flux = nn.functional.mse_loss(outputs["J_t"], J_obs)

    # ── 2. LRV prediction loss ─────────────────────────────────────────────────
    L_LRV = nn.functional.mse_loss(outputs["LRV"], LRV_obs)

    # ── 3. Physics constraint penalties ───────────────────────────────────────
    # relu(-x) > 0 only when x < 0, so this only penalises negative parameters
    # relu(Pc - 1) > 0 only when Pc > 1, penalising out-of-range probabilities
    params = outputs["params"]
    L_physics = (
        torch.relu(-params).sum()                             # all params >= 0
        + torch.relu(params[:, PARAM_IDX["Pc"]] - 1.0).sum() # Pc <= 1
    ) * lambda_physics

    # ── 4. Blocking regime classification loss ─────────────────────────────────
    # `torch.tensor(0.0)` creates a scalar zero tensor — the default if no labels
    L_regime = torch.tensor(0.0)
    if regime_labels is not None:
        # cross_entropy expects (B, n_classes) logits and (B,) integer labels
        L_regime = nn.functional.cross_entropy(outputs["regime_logits"], regime_labels)

    # ── 5. FedProx proximal term ───────────────────────────────────────────────
    L_fedprox = torch.tensor(0.0)
    if global_weights is not None and local_weights is not None:
        # Sum the squared L2 distance for each layer that exists in both dicts
        # `sum(... for k in ...)` is a generator expression — memory efficient
        prox = sum(
            torch.sum((local_weights[k] - global_weights[k]) ** 2)
            for k in global_weights
            if k in local_weights
        )
        L_fedprox = (fedprox_mu / 2.0) * prox

    # Total loss: sum of all four components
    return L_flux + L_LRV + L_physics + L_regime + L_fedprox
