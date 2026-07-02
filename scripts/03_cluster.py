"""Step 3: clustering on reduced descriptors and metrics collection."""
import gc
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import clustering, metrics, preprocess
from src.config import AVAILABLE_DESCRIPTORS, LABELS_DIR, RESULTS_DIR

# Maximum n_components per descriptor that fits 16 GB RAM.
N_COMPONENTS_MAP = {
    "efficientnet-b7": 1536,
    "osnet": 512,
    "vdc_type": 512,
    "vdc_color": 128,
}
N_CLUSTERS_LIST = [5, 10, 15]
SCALERS = ["standard", "minmax"]
METHODS = ["minibatch_kmeans", "kmeans"]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for descriptor in AVAILABLE_DESCRIPTORS:
        n_components = N_COMPONENTS_MAP[descriptor]
        for scaler in SCALERS:
            print(f"\n=== Loading {descriptor} / {scaler} / n_components={n_components} ===")
            X = preprocess.load_reduced(descriptor, scaler, n_components)

            for method in METHODS:
                for n_clusters in N_CLUSTERS_LIST:
                    print(f"  {method}, n_clusters={n_clusters}")
                    labels = clustering.cluster_descriptor(
                        X, method=method, n_clusters=n_clusters
                    )
                    scores = metrics.evaluate_clustering(X, labels)
                    records.append(
                        {
                            "descriptor": descriptor,
                            "scaler": scaler,
                            "n_components": n_components,
                            "method": method,
                            "n_clusters": n_clusters,
                            "calinski_harabasz": scores["calinski_harabasz"],
                            "davies_bouldin": scores["davies_bouldin"],
                            "n_noise": scores["n_noise"],
                        }
                    )
                    labels_path = (
                        LABELS_DIR
                        / f"labels_{descriptor}_{scaler}_pca{n_components}_{method}_{n_clusters}.joblib"
                    )
                    pd.DataFrame({"cluster": labels}).to_pickle(labels_path)
                    del labels
                    gc.collect()

            del X
            gc.collect()

    df = pd.DataFrame(records)
    summary_path = RESULTS_DIR / "clustering_summary_max_components.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(df.head(20))


if __name__ == "__main__":
    main()
