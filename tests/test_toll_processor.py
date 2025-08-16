# tests/test_toll_processor.py
"""Integration tests for TollProcessor."""
import pytest
from src.toll_processor import TollProcessor

def test_process_vehicle():
    """Test toll processing for a sample image."""
    processor = TollProcessor(db_path='data/database.db', toll_amount=10.0)
    # To be implemented: Test with mock recognizer and DB