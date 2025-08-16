# tests/test_recognizer.py
"""Unit tests for LicensePlateRecognizer."""
import pytest
from src.recognizer import LicensePlateRecognizer

def test_recognizer_init():
    """Test recognizer initialization."""
    recognizer = LicensePlateRecognizer(ocr_lang='en')
    assert recognizer is not None
    # To be implemented: Add more tests
