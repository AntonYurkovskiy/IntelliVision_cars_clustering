"""Step 5: save best clustering CSV and find outliers."""
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import load_data, outliers, preprocess, visualization
from src.config import LABELS_DIR, N_COMPONENTS, OUTLIERS_DIR, RESULTS_DIR

BEST_DESCRIPTOR = "vdc_type"
BEST_SCALER = "standard"
BEST_METHOD = "kmeans"
BEST_N_CLUSTERS = 5
OUTLIER_METHODS = ["isolation_forest", "local_outlier_factor"]
N_OUTLIER_FIGURES = 9


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTLIERS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Saving best clustering CSV ===")
    images_df = load_data.load_images_paths()
    images_df = images_df.rename(columns={images_df.columns[0]: "path"})
    labels = pd.read_pickle(
        LABELS_DIR / f"labels_{BEST_DESCRIPTOR}_{BEST_SCALER}_{BEST_METHOD}_{BEST_N_CLUSTERS}.joblib"
    )["cluster"].values
    images_df["cluster"] = labels
    best_csv = RESULTS_DIR / "best_clustering.csv"
    images_df.to_csv(best_csv, index=False)
    print(f"Saved: {best_csv}")

    print("\n=== Outlier detection ===")
    X = preprocess.load_reduced(BEST_DESCRIPTOR, BEST_SCALER, N_COMPONENTS)

    for method in OUTLIER_METHODS:
        print(f"\n  {method}")
        outlier_mask = outliers.find_outliers(X, method=method, contamination=0.01)
        n_outliers = int(outlier_mask.sum())
        print(f"  Found {n_outliers} outliers ({100 * n_outliers / len(outlier_mask):.2f}%)")

        outlier_df = images_df[outlier_mask].copy()
        outlier_path = OUTLIERS_DIR / f"outliers_{method}.csv"
        outlier_df.to_csv(outlier_path, index=False)
        print(f"  Saved: {outlier_path}")

        if n_outliers > 0:
            sample = outlier_df.head(N_OUTLIER_FIGURES)
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            axes = axes.flatten()
            for idx, ax in enumerate(axes):
                if idx < len(sample):
                    p = visualization._resolve_image_path(str(sample.iloc[idx]["path"]))
                    img = plt.imread(p)
                    ax.imshow(img)
                ax.axis("off")
            fig.suptitle(f"Outliers: {method}", fontsize=16)
            plt.tight_layout()
            fig_path = RESULTS_DIR / "figures" / f"outliers_{method}.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path, dpi=100)
            plt.close()
            print(f"  Saved figure: {fig_path}")

        del outlier_mask, outlier_df
        gc.collect()

    del X
    gc.collect()


if __name__ == "__main__":
    main()
