# tests/test_utils.py
"""Tests for utility functions."""
import pytest
from src.utils import load_image, PlateNotDetectedError

def test_load_image_invalid_path():
    """Test loading an invalid image path."""
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent.jpg")