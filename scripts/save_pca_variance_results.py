"""Save and plot already collected PCA variance results."""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
RESULTS_DIR = project_root / "results"
PLOT_DIR = RESULTS_DIR / "figures"


RESULTS = [
    {"descriptor": "osnet", "scaler": "standard", "n_components": 128, "explained_variance": 0.8748, "peak_mb": 2642.8, "time_s": 46.5},
    {"descriptor": "osnet", "scaler": "standard", "n_components": 256, "explained_variance": 0.9579, "peak_mb": 2897.5, "time_s": 37.8},
    {"descriptor": "osnet", "scaler": "standard", "n_components": 512, "explained_variance": 1.0000, "peak_mb": 4118.5, "time_s": 231.3},
    {"descriptor": "osnet", "scaler": "standard", "n_components": 1024, "explained_variance": 1.0000, "peak_mb": 4118.5, "time_s": 137.2},
    {"descriptor": "osnet", "scaler": "minmax", "n_components": 128, "explained_variance": 0.9146, "peak_mb": 2287.2, "time_s": 54.4},
    {"descriptor": "osnet", "scaler": "minmax", "n_components": 256, "explained_variance": 0.9682, "peak_mb": 2897.5, "time_s": 133.7},
    {"descriptor": "osnet", "scaler": "minmax", "n_components": 512, "explained_variance": 1.0000, "peak_mb": 4118.5, "time_s": 124.5},
    {"descriptor": "osnet", "scaler": "minmax", "n_components": 1024, "explained_variance": 1.0000, "peak_mb": 4118.5, "time_s": 90.7},
    {"descriptor": "efficientnet-b7", "scaler": "standard", "n_components": 128, "explained_variance": 0.5157, "peak_mb": 13213.4, "time_s": 108.1},
    {"descriptor": "efficientnet-b7", "scaler": "standard", "n_components": 256, "explained_variance": 0.6873, "peak_mb": 13213.4, "time_s": 108.5},
    {"descriptor": "efficientnet-b7", "scaler": "standard", "n_components": 512, "explained_variance": 0.8063, "peak_mb": 13213.4, "time_s": 141.1},
    {"descriptor": "efficientnet-b7", "scaler": "standard", "n_components": 1024, "explained_variance": 0.8740, "peak_mb": 13213.4, "time_s": 227.0},
    {"descriptor": "efficientnet-b7", "scaler": "minmax", "n_components": 128, "explained_variance": 0.5176, "peak_mb": 8793.2, "time_s": 97.3},
    {"descriptor": "efficientnet-b7", "scaler": "minmax", "n_components": 256, "explained_variance": 0.6898, "peak_mb": 9404.5, "time_s": 74.1},
    {"descriptor": "efficientnet-b7", "scaler": "minmax", "n_components": 512, "explained_variance": 0.8079, "peak_mb": 10627.5, "time_s": 126.0},
    {"descriptor": "efficientnet-b7", "scaler": "minmax", "n_components": 1024, "explained_variance": 0.8747, "peak_mb": 13074.8, "time_s": 239.3},
]


def save_results() -> None:
    """Save results to CSV and JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(RESULTS)
    csv_path = RESULTS_DIR / "pca_variance_analysis.csv"
    json_path = RESULTS_DIR / "pca_variance_analysis.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


def plot_results() -> None:
    """Plot explained variance vs n_components for each descriptor/scaler."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(RESULTS)

    plt.figure(figsize=(10, 6))
    for descriptor in df["descriptor"].unique():
        subset = df[df["descriptor"] == descriptor]
        for scaler in subset["scaler"].unique():
            sub = subset[subset["scaler"] == scaler].sort_values("n_components")
            label = f"{descriptor} / {scaler}"
            marker = "o" if scaler == "standard" else "s"
            plt.plot(
                sub["n_components"],
                sub["explained_variance"],
                marker=marker,
                label=label,
                linewidth=2,
            )

    plt.xlabel("n_components")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA explained variance vs n_components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    plot_path = PLOT_DIR / "pca_variance_vs_components.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {plot_path}")


def main():
    save_results()
    plot_results()


if __name__ == "__main__":
    main()
