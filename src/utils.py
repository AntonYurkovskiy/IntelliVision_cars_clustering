"""Utility helpers."""
import gc

import numpy as np


def free_memory(*variables):
    """Delete variables and run garbage collection."""
    for v in variables:
        del v
    gc.collect()


def memory_usage_mb(arr: np.ndarray) -> float:
    """Return memory usage of a numpy array in MB."""
    return arr.nbytes / (1024 ** 2)
