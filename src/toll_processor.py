# src/toll_processor.py
"""Core logic for toll processing: Recognize plate, deduct toll, log transaction."""
from .recognizer import LicensePlateRecognizer
from .database import DatabaseManager
import logging
from .utils import load_image

class TollProcessor:
    """Process tolls by recognizing plates and managing deductions."""
    
    def __init__(self, db_config, toll_amount, ocr_lang='en'):
        """Initialize with database config and recognizer."""
        self.recognizer = LicensePlateRecognizer(ocr_lang)
        self.db = DatabaseManager(db_config)
        self.toll_amount = toll_amount
    
    def process_vehicle(self, image_path):
        """Process an image to recognize plate and deduct toll.
        
        Returns a dict with keys: plate, vehicle_type, status ('success'|'failed'|'error'), message, new_balance (if success).
        """
        try:
            # load image to allow type detection and plate recognition
            image = load_image(image_path)
            if image is None:
                msg = f"Failed to load image: {image_path}"
                logging.error(msg)
                return {"plate": None, "vehicle_type": None, "status": "error", "message": msg}

            # detect vehicle type (no model paths here; can be extended)
            try:
                vehicle_type = self.recognizer.detect_vehicle_type(image)
            except Exception as e:
                logging.debug(f"Vehicle type detection error: {e}")
                vehicle_type = "unknown"

            # recognize plate text
            try:
                plate = self.recognizer.process_image(image_path)
            except Exception as e:
                msg = f"Plate recognition failed: {e}"
                logging.warning(msg)
                # log attempted transaction with unknown plate
                self.db.log_transaction("UNKNOWN", self.toll_amount, f"Failed: Plate recognition error ({vehicle_type})")
                return {"plate": None, "vehicle_type": vehicle_type, "status": "failed", "message": msg}

            if not plate:
                msg = "No plate text returned by recognizer."
                logging.warning(msg)
                self.db.log_transaction("UNKNOWN", self.toll_amount, f"Failed: No plate ({vehicle_type})")
                return {"plate": None, "vehicle_type": vehicle_type, "status": "failed", "message": msg}
            
            plate = plate.upper()
            try:
                new_balance = self.db.update_balance(plate, self.toll_amount)
                # include vehicle type in transaction status
                self.db.log_transaction(plate, self.toll_amount, f"Success ({vehicle_type})")
                msg = f"Toll of {self.toll_amount} deducted. New balance: {new_balance}"
                return {"plate": plate, "vehicle_type": vehicle_type, "status": "success", "message": msg, "new_balance": new_balance}
            except ValueError as ve:
                err_msg = str(ve)
                self.db.log_transaction(plate, self.toll_amount, f"Failed: {err_msg} ({vehicle_type})")
                logging.warning(f"Toll deduction failed for {plate}: {err_msg}")
                return {"plate": plate, "vehicle_type": vehicle_type, "status": "failed", "message": err_msg}
            except Exception as db_e:
                self.db.log_transaction(plate, self.toll_amount, f"Error: {db_e} ({vehicle_type})")
                logging.error(f"Unexpected DB error for {plate}: {db_e}")
                return {"plate": plate, "vehicle_type": vehicle_type, "status": "error", "message": str(db_e)}
        except Exception as e:
            logging.error(f"Error processing vehicle image {image_path}: {e}")
            return {"plate": None, "vehicle_type": None, "status": "error", "message": str(e)}