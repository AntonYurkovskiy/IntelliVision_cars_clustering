"""Preprocessing: scaling and dimensionality reduction."""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

from src.config import REDUCED_DIR, RANDOM_STATE


def scale_data(X: np.ndarray, method: str = "standard") -> np.ndarray:
    """Scale descriptor matrix."""
    if method == "standard":
        return StandardScaler().fit_transform(X)
    elif method == "minmax":
        return MinMaxScaler().fit_transform(X)
    elif method is None or method == "none":
        return X
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def reduce_dimension(
    X: np.ndarray,
    n_components: int,
    random_state: int = RANDOM_STATE,
    svd_solver: str = "randomized",
) -> tuple:
    """Reduce dimensionality with PCA.

    Returns:
        X_reduced, pca model, explained variance ratio sum.
    """
    n_components = min(n_components, X.shape[1])
    pca = PCA(
        n_components=n_components,
        random_state=random_state,
        svd_solver=svd_solver,
    )
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca, float(np.sum(pca.explained_variance_ratio_))


def get_reduced_path(descriptor_name: str, scaler: str, n_components: int) -> Path:
    """Path to save reduced descriptor."""
    REDUCED_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{descriptor_name}_{scaler}_pca{n_components}.joblib"
    return REDUCED_DIR / filename


def save_reduced(X_reduced: np.ndarray, descriptor_name: str, scaler: str, n_components: int) -> Path:
    """Save reduced descriptor to disk."""
    path = get_reduced_path(descriptor_name, scaler, n_components)
    joblib.dump(X_reduced, path, compress=3)
    return path


def load_reduced(descriptor_name: str, scaler: str, n_components: int) -> np.ndarray:
    """Load reduced descriptor from disk."""
    path = get_reduced_path(descriptor_name, scaler, n_components)
    if not path.exists():
        raise FileNotFoundError(f"Reduced file not found: {path}")
    return joblib.load(path)
