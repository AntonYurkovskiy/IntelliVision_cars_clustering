"""Tests for visualization module."""
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import visualization
from src.config import RAW_DATA_DIR

matplotlib.use("Agg")


@pytest.fixture
def tmp_image(tmp_path):
    """Create a tiny dummy PNG image for testing."""
    img_path = tmp_path / "dummy.png"
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    fig.savefig(img_path, dpi=10)
    plt.close(fig)
    return img_path


def test_resolve_image_path_absolute(tmp_path):
    p = tmp_path / "image.jpg"
    resolved = visualization._resolve_image_path(str(p))
    assert resolved == p


def test_resolve_image_path_relative():
    rel = "veriwild/0001/001.jpg"
    resolved = visualization._resolve_image_path(rel)
    assert resolved == RAW_DATA_DIR / rel


def test_resolve_image_path_normalizes_backslashes():
    resolved = visualization._resolve_image_path("veriwild\\0001\\001.jpg")
    assert resolved == RAW_DATA_DIR / "veriwild/0001/001.jpg"


def test_plot_tsne_runs_with_synthetic_data():
    X = np.random.randn(100, 4)
    labels = np.random.randint(0, 3, size=100)
    fig = plt.figure()
    visualization.plot_tsne(X, labels, n_samples=50, perplexity=5, random_state=0)
    plt.close(fig)


def test_plot_samples_images_empty_dataframe(capfd):
    df = pd.DataFrame({"paths": [], "cluster": []})
    visualization.plot_samples_images(df, cluster_label=0)
    out, _ = capfd.readouterr()
    assert "No images" in out


def test_plot_samples_images_with_dummy_image(tmp_image):
    df = pd.DataFrame({"paths": [str(tmp_image)], "cluster": [1]})
    visualization.plot_samples_images(df, cluster_label=1, nrows=1, ncols=1)
    plt.close("all")
