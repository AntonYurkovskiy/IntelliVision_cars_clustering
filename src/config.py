"""Project configuration and paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "IntelliVision_case"
DESCRIPTORS_DIR = DATA_DIR / "descriptors"
RAW_DATA_DIR = DATA_DIR / "raw_data"
IMAGES_PATHS_CSV = DATA_DIR / "images_paths.csv"

REDUCED_DIR = DATA_DIR / "reduced"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LABELS_DIR = RESULTS_DIR / "labels"
MODELS_DIR = RESULTS_DIR / "models"
OUTLIERS_DIR = RESULTS_DIR / "outliers"

DESCRIPTOR_NAMES = {
    "efficientnet-b7": "efficientnet-b7.pickle",
    "osnet": "osnet.pickle",
    "vdc_color": "mdc_color.pickle",
    "mdc_color": "mdc_color.pickle",
    "vdc_type": "mdc_type.pickle",
    "mdc_type": "mdc_type.pickle",
}

AVAILABLE_DESCRIPTORS = ["efficientnet-b7", "osnet", "vdc_color", "vdc_type"]

N_COMPONENTS = 128

# Maximum n_components per descriptor that fits 16 GB RAM.
N_COMPONENTS_MAP = {
    "efficientnet-b7": 1536,
    "osnet": 512,
    "vdc_type": 512,
    "vdc_color": 128,
}

RANDOM_STATE = 42
