"""Outlier detection methods."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

from src.config import RANDOM_STATE


def find_outliers(
    X: np.ndarray,
    method: str = "isolation_forest",
    **kwargs,
) -> np.ndarray:
    """Find outliers in a descriptor matrix.

    Returns:
        Boolean array: True for outliers, False for inliers.
    """
    method = method.lower().replace("-", "_")

    if method == "isolation_forest":
        contamination = kwargs.pop("contamination", 0.01)
        model = IsolationForest(
            contamination=contamination,
            random_state=kwargs.pop("random_state", RANDOM_STATE),
            n_estimators=kwargs.pop("n_estimators", 100),
            **kwargs,
        )
        labels = model.fit_predict(X)
        return labels == -1
    elif method == "local_outlier_factor":
        n_neighbors = kwargs.pop("n_neighbors", 20)
        contamination = kwargs.pop("contamination", 0.01)
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            **kwargs,
        )
        labels = model.fit_predict(X)
        return labels == -1
    elif method == "dbscan":
        eps = kwargs.pop("eps", 0.5)
        min_samples = kwargs.pop("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = model.fit_predict(X)
        return labels == -1
    else:
        raise ValueError(f"Unknown outlier method: {method}")
