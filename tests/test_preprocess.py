"""Tests for preprocess module."""
import numpy as np
import pytest

from src.preprocess import scale_data, reduce_dimension, get_reduced_path


RNG = np.random.RandomState(42)
X_SAMPLE = RNG.randn(100, 20).astype(np.float64)


class TestScaleData:
    def test_standard_zero_mean(self):
        X_scaled = scale_data(X_SAMPLE, method="standard")
        assert X_scaled.shape == X_SAMPLE.shape
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)

    def test_standard_unit_std(self):
        X_scaled = scale_data(X_SAMPLE, method="standard")
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)

    def test_minmax_range(self):
        X_scaled = scale_data(X_SAMPLE, method="minmax")
        assert X_scaled.shape == X_SAMPLE.shape
        assert X_scaled.min() >= 0.0 - 1e-10
        assert X_scaled.max() <= 1.0 + 1e-10

    def test_none_returns_original(self):
        X_scaled = scale_data(X_SAMPLE, method=None)
        np.testing.assert_array_equal(X_scaled, X_SAMPLE)

    def test_none_string_returns_original(self):
        X_scaled = scale_data(X_SAMPLE, method="none")
        np.testing.assert_array_equal(X_scaled, X_SAMPLE)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown scaling method"):
            scale_data(X_SAMPLE, method="l2")

    def test_preserves_shape(self):
        for method in ("standard", "minmax"):
            assert scale_data(X_SAMPLE, method=method).shape == X_SAMPLE.shape

    def test_default_method_is_standard(self):
        X_default = scale_data(X_SAMPLE)
        X_standard = scale_data(X_SAMPLE, method="standard")
        np.testing.assert_array_equal(X_default, X_standard)


class TestReduceDimension:
    def test_output_shape(self):
        X_red, pca, evr = reduce_dimension(X_SAMPLE, n_components=5)
        assert X_red.shape == (100, 5)

    def test_explained_variance_between_0_and_1(self):
        _, _, evr = reduce_dimension(X_SAMPLE, n_components=5)
        assert 0.0 < evr <= 1.0

    def test_clamps_n_components_to_n_features(self):
        X_red, pca, evr = reduce_dimension(X_SAMPLE, n_components=1000)
        assert X_red.shape[1] == X_SAMPLE.shape[1]

    def test_returns_three_values(self):
        result = reduce_dimension(X_SAMPLE, n_components=5)
        assert len(result) == 3

    def test_pca_model_has_correct_components(self):
        _, pca, _ = reduce_dimension(X_SAMPLE, n_components=8)
        assert pca.n_components_ == 8

    def test_full_variance_with_max_components(self):
        _, _, evr = reduce_dimension(X_SAMPLE, n_components=X_SAMPLE.shape[1])
        np.testing.assert_allclose(evr, 1.0, atol=1e-6)

    def test_reproducible_with_same_seed(self):
        X1, _, _ = reduce_dimension(X_SAMPLE, n_components=5, random_state=0)
        X2, _, _ = reduce_dimension(X_SAMPLE, n_components=5, random_state=0)
        np.testing.assert_array_equal(X1, X2)

    def test_single_component(self):
        X_red, _, _ = reduce_dimension(X_SAMPLE, n_components=1)
        assert X_red.shape == (100, 1)


class TestGetReducedPath:
    def test_returns_path_with_correct_filename(self):
        path = get_reduced_path("osnet", "standard", 64)
        assert path.name == "osnet_standard_pca64.joblib"

    def test_path_under_reduced_dir(self):
        from src.config import REDUCED_DIR
        path = get_reduced_path("osnet", "minmax", 32)
        assert path.parent == REDUCED_DIR

    def test_different_params_give_different_paths(self):
        p1 = get_reduced_path("osnet", "standard", 64)
        p2 = get_reduced_path("osnet", "minmax", 64)
        assert p1 != p2
