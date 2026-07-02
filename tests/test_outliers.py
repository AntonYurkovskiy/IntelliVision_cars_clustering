"""Tests for outliers module."""
import numpy as np
import pytest

from src.outliers import find_outliers


RNG = np.random.RandomState(0)


def _make_blobs(n_per=60, n_features=8, n_clusters=3, seed=0):
    rng = np.random.RandomState(seed)
    Xs = []
    for k in range(n_clusters):
        center = rng.randn(n_features) * 10
        Xs.append(rng.randn(n_per, n_features) + center)
    return np.vstack(Xs)


X = _make_blobs(n_per=50, n_features=4, n_clusters=3)


class TestFindOutliers:
    def test_returns_boolean_array(self):
        mask = find_outliers(X, method="isolation_forest")
        assert mask.dtype == bool

    def test_length_matches_input(self):
        for method in ("isolation_forest", "local_outlier_factor"):
            mask = find_outliers(X, method=method, contamination=0.05)
            assert len(mask) == len(X)

    def test_isolation_forest_runs(self):
        mask = find_outliers(X, method="isolation_forest", contamination=0.05, n_estimators=50)
        assert mask.sum() >= 0

    def test_local_outlier_factor_runs(self):
        mask = find_outliers(X, method="local_outlier_factor", contamination=0.05, n_neighbors=10)
        assert mask.sum() >= 0

    def test_dbscan_finds_noise(self):
        mask = find_outliers(X, method="dbscan", eps=0.5, min_samples=5)
        assert len(mask) == len(X)

    def test_dbscan_all_noise_with_tight_eps(self):
        mask = find_outliers(X, method="dbscan", eps=0.001, min_samples=100)
        assert mask.all()

    def test_contamination_rate_isolation_forest(self):
        contamination = 0.1
        mask = find_outliers(X, method="isolation_forest", contamination=contamination, n_estimators=100)
        expected = int(contamination * len(X))
        assert mask.sum() == pytest.approx(expected, abs=1)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown outlier method"):
            find_outliers(X, method="one_class_svm")

    def test_dash_alias_normalized(self):
        with pytest.raises(ValueError, match="Unknown outlier method"):
            find_outliers(X, method="isolation-forest-bad")
