"""Plot clustering metrics for max components results."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import FIGURES_DIR, RESULTS_DIR


def plot_individual(df: pd.DataFrame) -> None:
    """Plot CH and DB for individual descriptors."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    descriptors = df["descriptor"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Clustering metrics by descriptor (max components)")

    for ax, metric in zip(axes.flat, ["calinski_harabasz", "davies_bouldin"]):
        for descriptor in descriptors:
            subset = df[df["descriptor"] == descriptor]
            for scaler in subset["scaler"].unique():
                sub = subset[subset["scaler"] == scaler]
                for method in sub["method"].unique():
                    s = sub[sub["method"] == method].sort_values("n_clusters")
                    label = f"{descriptor} / {scaler} / {method}"
                    ax.plot(s["n_clusters"], s[metric], marker="o", label=label)

        ax.set_xlabel("n_clusters")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

    plot_path = FIGURES_DIR / "clustering_metrics_max_components.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")


def plot_combined(df: pd.DataFrame) -> None:
    """Plot CH and DB for combined descriptors."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Combined descriptors clustering metrics (max components)")

    for ax, metric in zip(axes, ["calinski_harabasz", "davies_bouldin"]):
        for combo in df["combination"].unique():
            for method in df["method"].unique():
                sub = df[
                    (df["combination"] == combo) & (df["method"] == method)
                ].sort_values("n_clusters")
                label = f"{combo} / {method}"
                ax.plot(sub["n_clusters"], sub[metric], marker="o", label=label)

        ax.set_xlabel("n_clusters")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plot_path = FIGURES_DIR / "combined_clustering_metrics_max_components.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")


def main():
    individual_path = RESULTS_DIR / "clustering_summary_max_components.csv"
    combined_path = RESULTS_DIR / "combined_clustering_summary_max_components.csv"

    df_individual = pd.read_csv(individual_path)
    df_combined = pd.read_csv(combined_path)

    plot_individual(df_individual)
    plot_combined(df_combined)


if __name__ == "__main__":
    main()
