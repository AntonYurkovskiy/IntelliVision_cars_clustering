"""Final verification script for the IntelliVision clustering project.

Checks that all expected artifacts and directories exist and reports
a concise summary of the project state.
"""
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    AVAILABLE_DESCRIPTORS,
    FIGURES_DIR,
    LABELS_DIR,
    MODELS_DIR,
    N_COMPONENTS_MAP,
    OUTLIERS_DIR,
    REDUCED_DIR,
    RESULTS_DIR,
)


def check_dir(path: Path, name: str) -> bool:
    if not path.exists():
        print(f"[MISSING] {name}: {path}")
        return False
    print(f"[OK] {name}: {path}")
    return True


def check_file(path: Path, name: str) -> bool:
    if not path.exists():
        print(f"[MISSING] {name}: {path}")
        return False
    print(f"[OK] {name}")
    return True


def main():
    all_ok = True

    print("=== Directory structure ===")
    for name, path in [
        ("results", RESULTS_DIR),
        ("figures", FIGURES_DIR),
        ("labels", LABELS_DIR),
        ("models", MODELS_DIR),
        ("outliers", OUTLIERS_DIR),
        ("reduced", REDUCED_DIR),
    ]:
        all_ok &= check_dir(path, name)

    print("\n=== Expected reports ===")
    all_ok &= check_file(RESULTS_DIR / "final_report.md", "final report")
    all_ok &= check_file(RESULTS_DIR / "cluster_descriptions.md", "cluster descriptions")
    all_ok &= check_file(RESULTS_DIR / "best_clustering.csv", "best clustering CSV")
    all_ok &= check_file(RESULTS_DIR / "clustering_summary_max_components.csv", "clustering summary max components")
    all_ok &= check_file(RESULTS_DIR / "combined_clustering_summary_max_components.csv", "combined clustering summary max components")

    print("\n=== Reduced descriptors ===")
    for descriptor in AVAILABLE_DESCRIPTORS:
        n_components = N_COMPONENTS_MAP[descriptor]
        for scaler in ("standard", "minmax"):
            path = REDUCED_DIR / f"{descriptor}_{scaler}_pca{n_components}.joblib"
            if not path.exists():
                print(f"[MISSING] {path.name}")
                all_ok = False
            else:
                print(f"[OK] {path.name}")

    print("\n=== Outliers ===")
    for method in ("isolation_forest", "local_outlier_factor"):
        path = OUTLIERS_DIR / f"outliers_{method}.csv"
        all_ok &= check_file(path, f"outliers {method}")

    print("\n=== Figures ===")
    required_figures = [
        "metrics_by_descriptor.png",
        "metrics_heatmap.png",
        "metrics_by_method_boxplot.png",
        "ch_vs_db_scatter.png",
    ]
    for name in required_figures:
        all_ok &= check_file(FIGURES_DIR / name, name)

    print("\n=== CSV summary preview ===")
    if (RESULTS_DIR / "clustering_summary.csv").exists():
        df = pd.read_csv(RESULTS_DIR / "clustering_summary.csv")
        print(f"clustering_summary.csv: {len(df)} rows, columns: {list(df.columns)}")
    if (RESULTS_DIR / "combined_clustering_summary.csv").exists():
        df = pd.read_csv(RESULTS_DIR / "combined_clustering_summary.csv")
        print(f"combined_clustering_summary.csv: {len(df)} rows, columns: {list(df.columns)}")

    print("\n" + "=" * 50)
    if all_ok:
        print("All expected artifacts are present.")
    else:
        print("Some artifacts are missing. See messages above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
