"""Clustering quality metrics."""
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute Calinski-Harabasz and Davies-Bouldin scores.

    Returns a dict with scores and the number of clusters (excluding noise).
    """
    labels = np.asarray(labels)
    mask = labels >= 0
    n_clusters = int(len(np.unique(labels[mask])))

    if n_clusters < 2:
        return {
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "n_clusters": n_clusters,
            "n_noise": int(np.sum(~mask)),
        }

    ch = calinski_harabasz_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    return {
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "n_clusters": n_clusters,
        "n_noise": int(np.sum(~mask)),
    }
