"""Tests for clustering module."""
import numpy as np
import pytest

from src.clustering import cluster_descriptor


RNG = np.random.RandomState(0)


def _blobs(n_per=60, n_features=8, n_clusters=3, seed=0):
    rng = np.random.RandomState(seed)
    Xs = []
    for k in range(n_clusters):
        center = rng.randn(n_features) * 10
        Xs.append(rng.randn(n_per, n_features) + center)
    return np.vstack(Xs)


X = _blobs()


class TestClusterDescriptorKMeans:
    def test_returns_array(self):
        labels = cluster_descriptor(X, method="kmeans", n_clusters=3)
        assert isinstance(labels, np.ndarray)

    def test_labels_length(self):
        labels = cluster_descriptor(X, method="kmeans", n_clusters=3)
        assert len(labels) == len(X)

    def test_correct_number_of_unique_labels(self):
        labels = cluster_descriptor(X, method="kmeans", n_clusters=3)
        assert len(np.unique(labels)) == 3

    def test_label_values_in_range(self):
        n_clusters = 4
        labels = cluster_descriptor(X, method="kmeans", n_clusters=n_clusters)
        assert labels.min() >= 0
        assert labels.max() < n_clusters

    def test_reproducible_with_same_seed(self):
        l1 = cluster_descriptor(X, method="kmeans", n_clusters=3, random_state=7)
        l2 = cluster_descriptor(X, method="kmeans", n_clusters=3, random_state=7)
        np.testing.assert_array_equal(l1, l2)

    def test_dash_alias_not_supported(self):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            cluster_descriptor(X, method="k-means", n_clusters=3)


class TestClusterDescriptorMiniBatchKMeans:
    def test_default_method_is_minibatch(self):
        labels = cluster_descriptor(X, n_clusters=3)
        assert len(labels) == len(X)

    def test_returns_correct_n_clusters(self):
        labels = cluster_descriptor(X, method="minibatch_kmeans", n_clusters=5)
        assert len(np.unique(labels)) == 5

    def test_custom_batch_size(self):
        labels = cluster_descriptor(X, method="minibatch_kmeans", n_clusters=3, batch_size=32)
        assert len(labels) == len(X)

    def test_labels_non_negative(self):
        labels = cluster_descriptor(X, method="minibatch_kmeans", n_clusters=3)
        assert labels.min() >= 0


class TestClusterDescriptorDBSCAN:
    def test_returns_array(self):
        labels = cluster_descriptor(X, method="dbscan", eps=3.0, min_samples=5)
        assert isinstance(labels, np.ndarray)

    def test_labels_length(self):
        labels = cluster_descriptor(X, method="dbscan", eps=3.0, min_samples=5)
        assert len(labels) == len(X)

    def test_noise_label_is_minus_one(self):
        labels = cluster_descriptor(X, method="dbscan", eps=0.001, min_samples=100)
        assert -1 in labels

    def test_tight_eps_finds_clusters(self):
        labels = cluster_descriptor(X, method="dbscan", eps=2.0, min_samples=3)
        unique = np.unique(labels)
        assert any(u >= 0 for u in unique)


class TestClusterDescriptorUnknownMethod:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            cluster_descriptor(X, method="spectral", n_clusters=3)

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            cluster_descriptor(X, method="", n_clusters=3)
