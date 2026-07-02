"""Tests for load_data module."""
import pytest

from src import load_data
from src.config import AVAILABLE_DESCRIPTORS, RAW_DATA_DIR


def test_raw_data_dir_exists():
    assert RAW_DATA_DIR.exists(), f"Raw data directory not found: {RAW_DATA_DIR}"


def test_get_descriptor_path():
    path = load_data.get_descriptor_path("efficientnet-b7")
    assert path.exists(), f"Descriptor path does not exist: {path}"


def test_load_images_paths():
    df = load_data.load_images_paths()
    assert len(df) > 0
    assert "path" in df.columns or "paths" in df.columns


def test_peek_descriptor():
    info = load_data.peek_descriptor("vdc_color")
    assert info["shape"][0] > 0
    assert info["shape"][1] > 0
