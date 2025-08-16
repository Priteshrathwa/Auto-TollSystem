# src/utils.py
"""Utility functions for logging, image loading, and error handling."""
import logging
import cv2
import os

# Custom exceptions
class PlateNotDetectedError(Exception):
    """Raised when license plate detection fails."""
    pass

class InsufficientBalanceError(Exception):
    """Raised when vehicle balance is insufficient."""
    pass

def setup_logging(log_file, log_level):
    """Set up logging to file and console."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_image(image_path):
    """Load an image using OpenCV."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.imread(image_path)