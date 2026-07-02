"""Step 4: analyze clustering summary and visualize best configurations."""
import gc
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import load_data, preprocess, visualization
from src.config import LABELS_DIR, N_COMPONENTS_MAP, RAW_DATA_DIR, RESULTS_DIR

N_TOP = 4
N_SAMPLES_TSNE = 5000
N_CLUSTERS_TO_PLOT = 5


def rank_configurations(df: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations by normalized composite score."""
    # Higher CH is better, lower DB is better
    df = df.copy()
    df["ch_norm"] = (df["calinski_harabasz"] - df["calinski_harabasz"].min()) / (
        df["calinski_harabasz"].max() - df["calinski_harabasz"].min() + 1e-9
    )
    df["db_norm"] = (df["davies_bouldin"].max() - df["davies_bouldin"]) / (
        df["davies_bouldin"].max() - df["davies_bouldin"].min() + 1e-9
    )
    df["score"] = df["ch_norm"] + df["db_norm"]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def main():
    summary_path = RESULTS_DIR / "clustering_summary_max_components.csv"
    df = pd.read_csv(summary_path)
    ranked = rank_configurations(df)

    print("=== Top configurations ===")
    print(ranked.head(N_TOP)[["descriptor", "scaler", "n_components", "method", "n_clusters", "calinski_harabasz", "davies_bouldin", "score"]])

    images_df = load_data.load_images_paths()
    images_df = images_df.rename(columns={images_df.columns[0]: "paths"})

    for i, row in ranked.head(N_TOP).iterrows():
        descriptor = row["descriptor"]
        scaler = row["scaler"]
        method = row["method"]
        n_clusters = int(row["n_clusters"])
        n_components = int(row["n_components"])

        print(f"\n=== Visualizing #{i+1}: {descriptor} / {scaler} / {method} / {n_clusters} clusters ===")

        X = preprocess.load_reduced(descriptor, scaler, n_components)
        labels_path = (
            LABELS_DIR
            / f"labels_{descriptor}_{scaler}_pca{n_components}_{method}_{n_clusters}.joblib"
        )
        labels = pd.read_pickle(labels_path)["cluster"].values

        data_with_labels = images_df.copy()
        data_with_labels["cluster"] = labels

        # t-SNE plot
        plt.figure(figsize=(12, 10))
        visualization.plot_tsne(
            X,
            labels,
            n_samples=N_SAMPLES_TSNE,
            title=f"{descriptor} / {scaler} / {method} / k={n_clusters}",
        )
        tsne_path = RESULTS_DIR / "figures" / f"tsne_{descriptor}_{scaler}_{method}_{n_clusters}.png"
        tsne_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(tsne_path, dpi=150)
        plt.close()
        print(f"  Saved t-SNE: {tsne_path}")

        # Sample images from top clusters
        for cluster_label in sorted(set(labels))[:N_CLUSTERS_TO_PLOT]:
            plt.figure(figsize=(10, 10))
            visualization.plot_samples_images(data_with_labels, cluster_label=cluster_label)
            img_path = (
                RESULTS_DIR
                / "figures"
                / f"images_{descriptor}_{scaler}_{method}_cluster{cluster_label}.png"
            )
            plt.savefig(img_path, dpi=100)
            plt.close()
            print(f"  Saved cluster {cluster_label} images: {img_path}")

        del X, labels, data_with_labels
        gc.collect()

    ranked.head(N_TOP).to_csv(RESULTS_DIR / "best_configurations.csv", index=False)
    print(f"\nBest configurations saved to {RESULTS_DIR / 'best_configurations.csv'}")


if __name__ == "__main__":
    main()
