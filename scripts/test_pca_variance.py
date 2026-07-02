"""Test PCA explained variance for osnet and efficientnet-b7 with different n_components."""
import sys
import gc
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import load_data
from src.config import RANDOM_STATE


# --- User-tunable parameters ---
# DESCRIPTORS = ["osnet", "efficientnet-b7"]
# SCALERS = ["standard", "minmax"]
# N_COMPONENTS_LIST = [128, 256, 512, 1024]

DESCRIPTORS = ["efficientnet-b7"]
SCALERS = ["standard", "minmax"]
N_COMPONENTS_LIST = [1536, 2048]

# --- Output paths ---
RESULTS_DIR = project_root / "results"
PLOT_DIR = RESULTS_DIR / "figures"


def _format_time(seconds: float) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _print_progress(current: int, total: int, start_time: float, width: int = 30) -> None:
    """Print a progress bar with elapsed time and ETA."""
    elapsed = time.time() - start_time
    if current > 0:
        eta = elapsed / current * (total - current)
    else:
        eta = 0.0
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    print(
        f"\r[{bar}] {current}/{total} | elapsed {_format_time(elapsed)} | ETA {_format_time(eta)}",
        end="",
        flush=True,
    )


def _print_progress_done() -> None:
    print()


def scale_data(X: np.ndarray, method: str):
    if method == "standard":
        return StandardScaler().fit_transform(X)
    elif method == "minmax":
        return MinMaxScaler().fit_transform(X)
    return X


def test_pca(descriptor_name: str, n_components: int, scaler: str):
    print(f"\n--- {descriptor_name} | {scaler} | n_components={n_components} ---")

    tracemalloc.start()
    t0 = time.time()

    X = load_data.load_descriptor(descriptor_name, dtype="float32")
    print(f"  Loaded: shape={X.shape}, memory={X.nbytes / 1024**2:.1f} MB")

    X_scaled = scale_data(X, scaler)
    del X
    gc.collect()

    n_components = min(n_components, X_scaled.shape[1])
    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE,
        svd_solver="randomized",
    )
    X_reduced = pca.fit_transform(X_scaled)
    explained_variance = float(np.sum(pca.explained_variance_ratio_))

    del X_scaled, X_reduced, pca
    gc.collect()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.time() - t0

    print(f"  Explained variance: {explained_variance:.4f}")
    print(f"  Peak memory: {peak / 1024**2:.1f} MB")
    print(f"  Time: {elapsed:.1f} s")

    return {
        "descriptor": descriptor_name,
        "scaler": scaler,
        "n_components": n_components,
        "explained_variance": explained_variance,
        "peak_mb": peak / 1024**2,
        "time_s": elapsed,
    }


def save_results(results: list) -> None:
    """Save results to CSV and JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "pca_variance_analysis.csv"
    json_path = RESULTS_DIR / "pca_variance_analysis.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


def plot_results(results: list) -> None:
    """Plot explained variance vs n_components for each descriptor/scaler."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

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

    print(f"  Plot: {plot_path}")


def main():
    total = len(DESCRIPTORS) * len(SCALERS) * len(N_COMPONENTS_LIST)
    print(f"=== PCA variance test: {total} experiments ===")
    start_time = time.time()

    results = []
    for descriptor in DESCRIPTORS:
        for scaler in SCALERS:
            for n in N_COMPONENTS_LIST:
                _print_progress(len(results) + 1, total, start_time)
                try:
                    result = test_pca(descriptor, n, scaler)
                    results.append(result)
                except Exception as e:
                    print(f"\nERROR: {descriptor}/{scaler}/{n}: {e}")

    _print_progress_done()
    elapsed = time.time() - start_time
    print(f"Total elapsed: {_format_time(elapsed)}")

    print("\n=== Summary ===")
    print(f"{'Descriptor':<20} {'Scaler':<10} {'n_comp':<8} {'Exp.Var':<10} {'Peak MB':<12} {'Time s':<8}")
    for r in results:
        print(
            f"{r['descriptor']:<20} {r['scaler']:<10} {r['n_components']:<8} "
            f"{r['explained_variance']:<10.4f} {r['peak_mb']:<12.1f} {r['time_s']:<8.1f}"
        )

    save_results(results)
    plot_results(results)


if __name__ == "__main__":
    main()
