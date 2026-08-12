"""Unit tests for shared/models/pinn.py — 100% coverage."""
import torch
import pytest

from shared.models.pinn import (
    INPUT_DIM,
    ParameterPredictor,
    BlockingRegimeClassifier,
    PhysicsSolver,
    FiltrationPINN,
    filtration_loss,
)
from shared.utils.constants import N_PARAMS, PARAM_IDX


B = 4    # batch size for all tests
T = 10   # time steps for all tests


def _features() -> torch.Tensor:
    return torch.rand(B, INPUT_DIM)


def _time() -> torch.Tensor:
    return torch.rand(B, T) * 60  # 0-60 minutes


# ── ParameterPredictor ─────────────────────────────────────────────────────────

class TestParameterPredictor:
    def test_output_shape(self) -> None:
        model = ParameterPredictor()
        params = model(_features())
        assert params.shape == (B, N_PARAMS)

    def test_all_non_pc_params_positive(self) -> None:
        model = ParameterPredictor()
        params = model(_features())
        for key in ("J0", "ks", "ki", "kc", "kcf", "k1", "k2", "Jcrit", "Dv"):
            assert (params[:, PARAM_IDX[key]] > 0).all(), f"{key} must be positive"

    def test_pc_in_open_zero_one(self) -> None:
        model = ParameterPredictor()
        Pc = model(_features())[:, PARAM_IDX["Pc"]]
        assert (Pc > 0).all() and (Pc < 1).all()

    def test_default_hidden_128(self) -> None:
        model = ParameterPredictor()
        assert isinstance(model, ParameterPredictor)

    def test_custom_hidden(self) -> None:
        model = ParameterPredictor(input_dim=INPUT_DIM, hidden=64)
        params = model(_features())
        assert params.shape == (B, N_PARAMS)

    def test_no_nan_in_output(self) -> None:
        model = ParameterPredictor()
        params = model(_features())
        assert not torch.isnan(params).any()


# ── BlockingRegimeClassifier ──────────────────────────────────────────────────

class TestBlockingRegimeClassifier:
    def test_output_shape(self) -> None:
        model = BlockingRegimeClassifier()
        logits = model(_features())
        assert logits.shape == (B, 5)

    def test_custom_input_dim(self) -> None:
        model = BlockingRegimeClassifier(input_dim=INPUT_DIM)
        logits = model(_features())
        assert logits.shape == (B, 5)

    def test_no_nan(self) -> None:
        logits = BlockingRegimeClassifier()(_features())
        assert not torch.isnan(logits).any()


# ── PhysicsSolver ─────────────────────────────────────────────────────────────

class TestPhysicsSolver:
    def _params(self) -> torch.Tensor:
        return ParameterPredictor()(_features())

    def test_j_t_shape(self) -> None:
        J_t, _ = PhysicsSolver()(self._params(), _time())
        assert J_t.shape == (B, T)

    def test_lrv_shape(self) -> None:
        _, LRV = PhysicsSolver()(self._params(), _time())
        assert LRV.shape == (B,)

    def test_j_t_positive(self) -> None:
        J_t, _ = PhysicsSolver()(self._params(), _time())
        assert (J_t > 0).all()

    def test_lrv_finite(self) -> None:
        _, LRV = PhysicsSolver()(self._params(), _time())
        assert torch.isfinite(LRV).all()


# ── FiltrationPINN ────────────────────────────────────────────────────────────

class TestFiltrationPINN:
    def test_output_keys(self) -> None:
        model = FiltrationPINN()
        out = model(_features(), _time())
        assert set(out.keys()) == {"J_t", "LRV", "params", "regime_logits"}

    def test_j_t_shape(self) -> None:
        out = FiltrationPINN()(_features(), _time())
        assert out["J_t"].shape == (B, T)

    def test_lrv_shape(self) -> None:
        out = FiltrationPINN()(_features(), _time())
        assert out["LRV"].shape == (B,)

    def test_params_shape(self) -> None:
        out = FiltrationPINN()(_features(), _time())
        assert out["params"].shape == (B, N_PARAMS)

    def test_regime_logits_shape(self) -> None:
        out = FiltrationPINN()(_features(), _time())
        assert out["regime_logits"].shape == (B, 5)

    def test_custom_hidden(self) -> None:
        model = FiltrationPINN(input_dim=INPUT_DIM, hidden=64)
        out = model(_features(), _time())
        assert out["J_t"].shape == (B, T)


# ── filtration_loss ───────────────────────────────────────────────────────────

class TestFiltrationLoss:
    def _outputs(self) -> dict:
        return FiltrationPINN()(_features(), _time())

    def test_returns_scalar_tensor(self) -> None:
        loss = filtration_loss(
            self._outputs(),
            J_obs=torch.rand(B, T),
            LRV_obs=torch.rand(B),
            regime_labels=None,
            global_weights=None,
            local_weights=None,
        )
        assert loss.shape == ()

    def test_loss_nonneg(self) -> None:
        loss = filtration_loss(
            self._outputs(), torch.rand(B, T), torch.rand(B),
            None, None, None,
        )
        assert loss.item() >= 0

    def test_loss_finite(self) -> None:
        loss = filtration_loss(
            self._outputs(), torch.rand(B, T), torch.rand(B),
            None, None, None,
        )
        assert torch.isfinite(loss)

    def test_with_regime_labels(self) -> None:
        labels = torch.randint(0, 5, (B,))
        loss = filtration_loss(
            self._outputs(), torch.rand(B, T), torch.rand(B),
            regime_labels=labels, global_weights=None, local_weights=None,
        )
        assert torch.isfinite(loss)

    def test_without_regime_labels(self) -> None:
        loss = filtration_loss(
            self._outputs(), torch.rand(B, T), torch.rand(B),
            regime_labels=None, global_weights=None, local_weights=None,
        )
        assert torch.isfinite(loss)

    def test_with_fedprox(self) -> None:
        model = FiltrationPINN()
        out = model(_features(), _time())
        local_w = {k: v.detach().clone() for k, v in model.predictor.named_parameters()}
        global_w = {k: torch.randn_like(local_w[k]) for k in local_w}
        loss = filtration_loss(
            out, torch.rand(B, T), torch.rand(B),
            None, global_w, local_w, fedprox_mu=0.01,
        )
        assert torch.isfinite(loss)

    def test_fedprox_layer_not_in_local_skipped(self) -> None:
        """A global layer absent from local must not raise KeyError."""
        out = self._outputs()
        global_w = {"ghost_layer": torch.tensor([1.0, 2.0])}
        local_w: dict = {}
        loss = filtration_loss(
            out, torch.rand(B, T), torch.rand(B),
            None, global_w, local_w,
        )
        assert torch.isfinite(loss)

    def test_custom_lambda_physics(self) -> None:
        loss = filtration_loss(
            self._outputs(), torch.rand(B, T), torch.rand(B),
            None, None, None, lambda_physics=5.0,
        )
        assert torch.isfinite(loss)

    def test_no_fedprox_when_global_is_none(self) -> None:
        """global_weights=None → L_fedprox=0, still valid loss."""
        model = FiltrationPINN()
        out = model(_features(), _time())
        local_w = {k: v.detach().clone() for k, v in model.predictor.named_parameters()}
        loss = filtration_loss(
            out, torch.rand(B, T), torch.rand(B),
            None, global_weights=None, local_weights=local_w,
        )
        assert torch.isfinite(loss)

    def test_no_fedprox_when_local_is_none(self) -> None:
        """local_weights=None → L_fedprox=0."""
        global_w = {"layer": torch.rand(3)}
        loss = filtration_loss(
            self._outputs(), torch.rand(B, T), torch.rand(B),
            None, global_weights=global_w, local_weights=None,
        )
        assert torch.isfinite(loss)
