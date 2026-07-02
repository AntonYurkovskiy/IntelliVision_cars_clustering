"""Step 1: explore descriptor sizes and image paths."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import load_data
from src.config import AVAILABLE_DESCRIPTORS, RAW_DATA_DIR


def resolve_image_path(rel_path: str) -> Path:
    """Build absolute path from relative path in CSV."""
    rel_path = rel_path.strip().replace("\\", "/")
    return RAW_DATA_DIR / rel_path


def main():
    print("=== IntelliVision data exploration ===\n")

    for name in AVAILABLE_DESCRIPTORS:
        info = load_data.peek_descriptor(name)
        print(
            f"{info['name']}: shape={info['shape']}, "
            f"dtype={info['dtype']}, memory={info['memory_mb']:.1f} MB"
        )

    print("\n=== Image paths CSV ===")
    df = load_data.load_images_paths()
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    missing = 0
    sample_paths = df.iloc[:10, 0].tolist()
    for p in sample_paths:
        abs_p = resolve_image_path(p)
        if not abs_p.exists():
            missing += 1
            print(f"MISSING: {abs_p}")
    print(f"Sample path existence check: {10 - missing}/10 exist")


if __name__ == "__main__":
    main()
