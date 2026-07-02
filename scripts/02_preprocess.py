"""Step 2: scale and reduce dimensionality for each descriptor."""
import gc
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import load_data, preprocess
from src.config import AVAILABLE_DESCRIPTORS, N_COMPONENTS_MAP
SCALERS = ["standard", "minmax"]


def main():
    for name in AVAILABLE_DESCRIPTORS:
        n_components = N_COMPONENTS_MAP[name]
        for scaler in SCALERS:
            print(f"\n=== Processing {name} with {scaler} scaler ===")
            X = load_data.load_descriptor(name, dtype="float32")
            print(f"Loaded shape={X.shape}, memory={X.nbytes / 1024**2:.1f} MB")

            X_scaled = preprocess.scale_data(X, method=scaler)
            del X
            gc.collect()

            X_reduced, pca, variance = preprocess.reduce_dimension(
                X_scaled, n_components=n_components
            )
            del X_scaled
            gc.collect()

            path = preprocess.save_reduced(X_reduced, name, scaler, n_components)
            print(
                f"Reduced shape={X_reduced.shape}, "
                f"explained_variance={variance:.4f}, saved to {path}"
            )

            del X_reduced, pca
            gc.collect()


if __name__ == "__main__":
    main()
