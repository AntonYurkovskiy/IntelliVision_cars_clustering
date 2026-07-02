"""Load descriptors and image paths."""
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DESCRIPTORS_DIR, IMAGES_PATHS_CSV, DESCRIPTOR_NAMES, AVAILABLE_DESCRIPTORS


def get_descriptor_path(name: str) -> Path:
    """Return the path to a descriptor pickle file."""
    if name not in DESCRIPTOR_NAMES:
        raise ValueError(f"Unknown descriptor: {name}. Available: {AVAILABLE_DESCRIPTORS}")
    return DESCRIPTORS_DIR / DESCRIPTOR_NAMES[name]


def load_descriptor(name: str, dtype: str = "float32") -> np.ndarray:
    """Load a descriptor pickle file as a numpy array of the requested dtype."""
    path = get_descriptor_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Descriptor file not found: {path}")
    with open(path, "rb") as f:
        X = pickle.load(f)
    X = np.asarray(X, dtype=dtype)
    return X


def load_images_paths() -> pd.DataFrame:
    """Load the CSV with image paths."""
    if not IMAGES_PATHS_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {IMAGES_PATHS_CSV}")
    df = pd.read_csv(IMAGES_PATHS_CSV)
    return df


def peek_descriptor(name: str) -> dict:
    """Load descriptor metadata without keeping it in memory."""
    X = load_descriptor(name, dtype="float32")
    info = {
        "name": name,
        "shape": X.shape,
        "dtype": str(X.dtype),
        "memory_mb": X.nbytes / (1024 ** 2),
    }
    del X
    gc.collect()
    return info
