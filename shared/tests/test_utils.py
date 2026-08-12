"""Unit tests for shared/utils — constants and logging_config. 100% coverage."""
import structlog

from shared.utils.constants import (
    J_MIN, J_MAX,
    KS_BOUNDS, KI_BOUNDS, KC_BOUNDS, KCF_BOUNDS, K1_BOUNDS, K2_BOUNDS,
    PC_BOUNDS, JCRIT_BOUNDS, LAMBDA_BOUNDS,
    LRV_MIN_PARVOVIRUS, LRV_MIN_RETROVIRUS, LRV_MIN_HERPESVIRUS,
    FLUX_RATIO_MIN, PARAM_IDX, N_PARAMS, HERMIA_MODELS,
)
from shared.utils.logging_config import configure_logging, get_logger


class TestConstants:
    def test_flux_bounds(self) -> None:
        assert J_MIN == 0.1
        assert J_MAX == 500.0

    def test_ks_bounds(self) -> None:
        assert KS_BOUNDS == (0.0, 1e3)

    def test_ki_bounds(self) -> None:
        assert KI_BOUNDS == (0.0, 1e3)

    def test_kc_bounds(self) -> None:
        assert KC_BOUNDS == (0.0, 1e3)

    def test_kcf_bounds(self) -> None:
        assert KCF_BOUNDS == (0.0, 1e3)

    def test_k1_bounds(self) -> None:
        assert K1_BOUNDS == (0.0, 1e2)

    def test_k2_bounds(self) -> None:
        assert K2_BOUNDS == (0.0, 1e2)

    def test_pc_bounds(self) -> None:
        assert PC_BOUNDS == (0.0, 1.0)

    def test_jcrit_bounds(self) -> None:
        assert JCRIT_BOUNDS == (1.0, 500.0)

    def test_lambda_bounds(self) -> None:
        assert LAMBDA_BOUNDS == (0.0, 100.0)

    def test_lrv_min_parvovirus(self) -> None:
        assert LRV_MIN_PARVOVIRUS == 4.0

    def test_lrv_min_retrovirus(self) -> None:
        assert LRV_MIN_RETROVIRUS == 4.0

    def test_lrv_min_herpesvirus(self) -> None:
        assert LRV_MIN_HERPESVIRUS == 4.0

    def test_flux_ratio_min(self) -> None:
        assert FLUX_RATIO_MIN == 0.2

    def test_param_idx_keys(self) -> None:
        expected = {"J0", "ks", "ki", "kc", "kcf", "k1", "k2", "Pc", "Jcrit", "Dv"}
        assert set(PARAM_IDX.keys()) == expected

    def test_param_idx_unique_values(self) -> None:
        assert len(set(PARAM_IDX.values())) == len(PARAM_IDX)

    def test_param_idx_j0_is_zero(self) -> None:
        assert PARAM_IDX["J0"] == 0

    def test_n_params_is_ten(self) -> None:
        assert N_PARAMS == 10

    def test_n_params_equals_param_idx_length(self) -> None:
        assert N_PARAMS == len(PARAM_IDX)

    def test_hermia_models_list(self) -> None:
        assert HERMIA_MODELS == [
            "standard", "complete", "intermediate", "cake", "combined_1a"
        ]

    def test_hermia_models_count(self) -> None:
        assert len(HERMIA_MODELS) == 5


class TestLoggingConfig:
    def test_configure_logging_default(self) -> None:
        configure_logging()  # should not raise

    def test_configure_logging_with_debug_level(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        configure_logging()

    def test_configure_logging_with_warning_level(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        configure_logging()

    def test_get_logger_returns_object(self) -> None:
        log = get_logger("test.module")
        assert log is not None

    def test_get_logger_different_names(self) -> None:
        log_a = get_logger("module.a")
        log_b = get_logger("module.b")
        assert log_a is not None
        assert log_b is not None

    def test_get_logger_is_callable(self) -> None:
        log = get_logger("some.module")
        assert callable(getattr(log, "info", None))
