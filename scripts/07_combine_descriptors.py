"""Step 7: combine descriptors and cluster them."""
import gc
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import metrics, preprocess
from src.config import LABELS_DIR, MODELS_DIR, N_COMPONENTS_MAP, RANDOM_STATE, RESULTS_DIR
COMBINATIONS = [
    ("vdc_type", "vdc_color"),
    ("vdc_type", "osnet"),
]
SCALER = "standard"
N_CLUSTERS = [5, 10, 15]
METHODS = ["minibatch_kmeans", "kmeans"]


def concatenate_reduced(descriptors, scaler, n_components_map):
    """Load and horizontally concatenate reduced descriptors."""
    parts = [
        preprocess.load_reduced(d, scaler, n_components_map[d]) for d in descriptors
    ]
    combined = pd.concat([pd.DataFrame(p) for p in parts], axis=1)
    return combined.values.astype("float32")


def run_clustering(X, method, n_clusters):
    if method == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init="auto",
            batch_size=2048,
        )
    elif method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init="auto",
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    labels = model.fit_predict(X)
    return labels, model


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for combo in COMBINATIONS:
        combo_name = "+".join(combo)
        print(f"\n=== Combination: {combo_name} ===")
        X = concatenate_reduced(combo, SCALER, N_COMPONENTS_MAP)
        print(f"  Combined shape: {X.shape}")

        for method in METHODS:
            for n_clusters in N_CLUSTERS:
                print(f"  {method}, n_clusters={n_clusters}")
                labels, model = run_clustering(X, method, n_clusters)
                scores = metrics.evaluate_clustering(X, labels)
                ch = scores["calinski_harabasz"]
                db = scores["davies_bouldin"]

                summary_rows.append(
                    {
                        "combination": combo_name,
                        "scaler": SCALER,
                        "method": method,
                        "n_clusters": n_clusters,
                        "calinski_harabasz": ch,
                        "davies_bouldin": db,
                        "n_noise": scores["n_noise"],
                    }
                )

                labels_path = (
                    LABELS_DIR
                    / f"labels_combined_{combo_name}_{SCALER}_{method}_{n_clusters}.joblib"
                )
                pd.DataFrame({"cluster": labels}).to_pickle(labels_path)

                joblib.dump(
                    model, MODELS_DIR / f"model_combined_{combo_name}_{SCALER}_{method}_{n_clusters}.joblib"
                )

                del labels, model
                gc.collect()

        del X
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "combined_clustering_summary_max_components.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCombined clustering summary saved to {summary_path}")

    best = summary_df.loc[summary_df["calinski_harabasz"].idxmax()]
    print("\nBest combined configuration:")
    print(best)


if __name__ == "__main__":
    main()
