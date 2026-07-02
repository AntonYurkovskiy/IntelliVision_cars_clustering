"""Clustering algorithms."""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN

from src.config import RANDOM_STATE


def cluster_descriptor(
    X: np.ndarray,
    method: str = "minibatch_kmeans",
    n_clusters: int = 10,
    random_state: int = RANDOM_STATE,
    **kwargs,
) -> np.ndarray:
    """Cluster a descriptor matrix.

    Supported methods:
        - "kmeans": KMeans
        - "minibatch_kmeans": MiniBatchKMeans (default, memory efficient)
        - "dbscan": DBSCAN (labels -1 are noise)
    """
    method = method.lower().replace("-", "_")

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    elif method == "minibatch_kmeans":
        batch_size = kwargs.pop("batch_size", 1024)
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            n_init="auto",
            **kwargs,
        )
    elif method == "dbscan":
        eps = kwargs.pop("eps", 0.5)
        min_samples = kwargs.pop("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = model.fit_predict(X)
    return labels
