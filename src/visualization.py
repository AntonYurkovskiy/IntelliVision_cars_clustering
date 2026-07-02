"""Visualization helpers for clusters and images."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from src.config import RAW_DATA_DIR


def _resolve_image_path(path: str) -> Path:
    """Make image path absolute using RAW_DATA_DIR."""
    path = path.strip().replace("\\", "/")
    p = Path(path)
    if p.is_absolute():
        return p
    return RAW_DATA_DIR / path


def plot_samples_images(
    data: pd.DataFrame,
    cluster_label: int,
    nrows: int = 3,
    ncols: int = 3,
    figsize: tuple = (12, 12),
    random_state: int = 42,
) -> None:
    """Plot a grid of random images from a given cluster.

    Args:
        data: DataFrame with columns 'paths' and 'cluster'.
        cluster_label: cluster label to visualize.
    """
    rng = np.random.default_rng(random_state)
    samples = data[data["cluster"] == cluster_label]
    if samples.empty:
        print(f"No images for cluster {cluster_label}")
        return

    sample_indices = rng.choice(samples.index, size=min(nrows * ncols, len(samples)), replace=False)
    paths = samples.loc[sample_indices, "paths"]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for idx, ax in enumerate(axes.flat):
        if idx < len(paths):
            path = _resolve_image_path(str(paths.iloc[idx]))
            img = plt.imread(path)
            ax.imshow(img)
        ax.axis("off")
    fig.suptitle(f"Images from cluster {cluster_label}", fontsize=16)
    plt.tight_layout()


def plot_tsne(
    X: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 5000,
    perplexity: int = 30,
    random_state: int = 42,
    title: str = "t-SNE visualization",
) -> None:
    """Plot 2D t-SNE of a sample of points colored by cluster labels."""
    n_samples = min(n_samples, X.shape[0])
    indices = np.random.default_rng(random_state).choice(X.shape[0], size=n_samples, replace=False)
    X_sample = X[indices]
    labels_sample = labels[indices]

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
    X_2d = tsne.fit_transform(X_sample)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_sample, cmap="tab20", s=5, alpha=0.7)
    plt.colorbar(scatter, label="cluster")
    plt.title(title)
    plt.tight_layout()
