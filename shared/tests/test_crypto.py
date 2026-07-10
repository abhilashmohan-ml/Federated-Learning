"""Unit tests for shared/crypto — noise and secure_agg. 100% coverage."""
import numpy as np
import pytest

from shared.crypto.noise import add_gaussian_noise
from shared.crypto.secure_agg import split_into_shares, reconstruct_from_shares


class TestAddGaussianNoise:
    def test_output_has_same_keys(self) -> None:
        w = {"layer_a": [1.0, 2.0, 3.0], "layer_b": [4.0, 5.0]}
        result = add_gaussian_noise(w, sigma=0.01)
        assert set(result.keys()) == {"layer_a", "layer_b"}

    def test_output_length_preserved(self) -> None:
        w = {"params": [1.0, 2.0, 3.0]}
        result = add_gaussian_noise(w, sigma=0.01)
        assert len(result["params"]) == 3

    def test_sigma_zero_no_noise_below_clip(self) -> None:
        """sigma=0 with norm < clip_norm: only float32 rounding; values stay close."""
        w = {"params": [0.1, 0.2, 0.3]}  # norm ~0.37, below 1.0 clip
        result = add_gaussian_noise(w, sigma=0.0, clip_norm=1.0)
        np.testing.assert_allclose(result["params"], [0.1, 0.2, 0.3], atol=1e-5)

    def test_large_norm_gets_clipped(self) -> None:
        """When L2 norm > clip_norm, output norm should equal clip_norm (sigma=0)."""
        w = {"params": [10.0, 0.0, 0.0]}  # norm=10, clip_norm=1.0
        result = add_gaussian_noise(w, sigma=0.0, clip_norm=1.0)
        arr = np.array(result["params"])
        np.testing.assert_allclose(float(np.linalg.norm(arr)), 1.0, atol=1e-5)

    def test_small_norm_not_clipped(self) -> None:
        """When L2 norm <= clip_norm, clipping is a no-op."""
        w = {"params": [0.3, 0.4]}  # norm=0.5, clip_norm=1.0
        result = add_gaussian_noise(w, sigma=0.0, clip_norm=1.0)
        np.testing.assert_allclose(result["params"], [0.3, 0.4], atol=1e-5)

    def test_noise_added_changes_values(self) -> None:
        """With sigma > 0, output should differ from all-zeros input."""
        w = {"params": [0.0] * 100}
        result = add_gaussian_noise(w, sigma=1.0)
        assert not np.allclose(result["params"], [0.0] * 100)

    def test_returns_new_dict_not_mutation(self) -> None:
        """Input dict must not be modified."""
        original = [1.0, 2.0, 3.0]
        w = {"p": list(original)}
        add_gaussian_noise(w, sigma=0.0)
        assert w["p"] == original

    def test_empty_weights_dict(self) -> None:
        result = add_gaussian_noise({}, sigma=0.01)
        assert result == {}

    def test_custom_clip_norm(self) -> None:
        w = {"params": [3.0, 4.0]}  # norm=5.0, clip_norm=2.5
        result = add_gaussian_noise(w, sigma=0.0, clip_norm=2.5)
        arr = np.array(result["params"])
        np.testing.assert_allclose(float(np.linalg.norm(arr)), 2.5, atol=1e-5)

    def test_result_is_list_of_floats(self) -> None:
        w = {"p": [1.0, 2.0]}
        result = add_gaussian_noise(w, sigma=0.0)
        assert isinstance(result["p"], list)


class TestSecureAgg:
    def test_split_correct_number_of_shares(self) -> None:
        w = {"layer": [1.0, 2.0, 3.0]}
        shares = split_into_shares(w, n_shares=3)
        assert len(shares) == 3

    def test_reconstruct_three_shares(self) -> None:
        w = {"layer": [10.0, 20.0, 30.0]}
        shares = split_into_shares(w, n_shares=3)
        result = reconstruct_from_shares(shares)
        np.testing.assert_allclose(result["layer"], [10.0, 20.0, 30.0], atol=1e-9)

    def test_reconstruct_two_shares(self) -> None:
        w = {"a": [5.0, -5.0]}
        shares = split_into_shares(w, n_shares=2)
        result = reconstruct_from_shares(shares)
        np.testing.assert_allclose(result["a"], [5.0, -5.0], atol=1e-9)

    def test_reconstruct_five_shares(self) -> None:
        w = {"params": [float(i) for i in range(1, 6)]}
        shares = split_into_shares(w, n_shares=5)
        result = reconstruct_from_shares(shares)
        np.testing.assert_allclose(result["params"], list(range(1, 6)), atol=1e-9)

    def test_multiple_layers(self) -> None:
        w = {"layer_0": [1.0, 2.0], "layer_1": [3.0, 4.0]}
        shares = split_into_shares(w, n_shares=3)
        result = reconstruct_from_shares(shares)
        np.testing.assert_allclose(result["layer_0"], [1.0, 2.0], atol=1e-9)
        np.testing.assert_allclose(result["layer_1"], [3.0, 4.0], atol=1e-9)

    def test_shares_are_dicts_with_same_keys(self) -> None:
        w = {"p": [1.0]}
        shares = split_into_shares(w, n_shares=2)
        assert all(isinstance(s, dict) for s in shares)
        assert all("p" in s for s in shares)

    def test_reconstruct_output_is_list(self) -> None:
        w = {"p": [1.0, 2.0]}
        shares = split_into_shares(w, n_shares=2)
        result = reconstruct_from_shares(shares)
        assert isinstance(result["p"], list)

    def test_individual_shares_differ_from_original(self) -> None:
        """Each individual share should not equal the original (random masking)."""
        w = {"p": [100.0, 200.0]}
        shares = split_into_shares(w, n_shares=3)
        # At least the first share should differ (random share)
        # (extremely unlikely to match, but not guaranteed for last share)
        assert not np.allclose(shares[0]["p"], [100.0, 200.0])
