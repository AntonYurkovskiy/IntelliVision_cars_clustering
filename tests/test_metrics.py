"""Tests for metrics module."""
import numpy as np
import pytest

from src.metrics import evaluate_clustering


def _make_blobs(n_per_cluster=50, n_features=10, n_clusters=3, seed=0):
    rng = np.random.RandomState(seed)
    Xs, ys = [], []
    for k in range(n_clusters):
        center = rng.randn(n_features) * 5
        Xs.append(rng.randn(n_per_cluster, n_features) + center)
        ys.extend([k] * n_per_cluster)
    return np.vstack(Xs), np.array(ys)


class TestEvaluateClustering:
    def test_returns_dict_with_expected_keys(self):
        X, labels = _make_blobs()
        result = evaluate_clustering(X, labels)
        assert set(result.keys()) == {"calinski_harabasz", "davies_bouldin", "n_clusters", "n_noise"}

    def test_valid_clustering_finite_scores(self):
        X, labels = _make_blobs()
        result = evaluate_clustering(X, labels)
        assert np.isfinite(result["calinski_harabasz"])
        assert np.isfinite(result["davies_bouldin"])

    def test_n_clusters_correct(self):
        X, labels = _make_blobs(n_clusters=4)
        result = evaluate_clustering(X, labels)
        assert result["n_clusters"] == 4

    def test_no_noise_points(self):
        X, labels = _make_blobs()
        result = evaluate_clustering(X, labels)
        assert result["n_noise"] == 0

    def test_noise_points_counted(self):
        X, labels = _make_blobs(n_per_cluster=30, n_clusters=3)
        noisy_labels = labels.copy().astype(int)
        noisy_labels[:5] = -1
        result = evaluate_clustering(X, noisy_labels)
        assert result["n_noise"] == 5

    def test_noise_points_excluded_from_scoring(self):
        X, labels = _make_blobs(n_per_cluster=50, n_clusters=3)
        noisy_labels = labels.copy().astype(int)
        noisy_labels[:10] = -1
        result_clean = evaluate_clustering(X, labels)
        result_noisy = evaluate_clustering(X, noisy_labels)
        assert result_noisy["n_clusters"] == 3
        assert np.isfinite(result_noisy["calinski_harabasz"])

    def test_single_cluster_returns_nan(self):
        X = np.random.randn(100, 5)
        labels = np.zeros(100, dtype=int)
        result = evaluate_clustering(X, labels)
        assert result["n_clusters"] == 1
        assert np.isnan(result["calinski_harabasz"])
        assert np.isnan(result["davies_bouldin"])

    def test_all_noise_returns_nan(self):
        X = np.random.randn(50, 5)
        labels = np.full(50, -1, dtype=int)
        result = evaluate_clustering(X, labels)
        assert result["n_clusters"] == 0
        assert np.isnan(result["calinski_harabasz"])
        assert np.isnan(result["davies_bouldin"])
        assert result["n_noise"] == 50

    def test_calinski_harabasz_positive(self):
        X, labels = _make_blobs()
        result = evaluate_clustering(X, labels)
        assert result["calinski_harabasz"] > 0

    def test_davies_bouldin_non_negative(self):
        X, labels = _make_blobs()
        result = evaluate_clustering(X, labels)
        assert result["davies_bouldin"] >= 0

    def test_labels_as_list(self):
        X, labels = _make_blobs()
        result = evaluate_clustering(X, list(labels))
        assert np.isfinite(result["calinski_harabasz"])

    def test_two_clusters(self):
        X, labels = _make_blobs(n_clusters=2)
        result = evaluate_clustering(X, labels)
        assert result["n_clusters"] == 2
        assert np.isfinite(result["calinski_harabasz"])
