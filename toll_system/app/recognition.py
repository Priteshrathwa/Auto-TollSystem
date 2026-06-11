import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import re
import os
# Add PIL import for better image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Some image processing features may be limited.")

class PlateRecognition:
    def __init__(self):
        # Lazy load the model to speed up initialization
        self._model = None
        self.model_path = r"D:\AUTOTOLL\yolo model\best.pt"
        try:
            self.reader = easyocr.Reader(['en'])
            self.ocr_available = True
        except Exception as e:
            print(f"Warning: EasyOCR initialization failed: {e}")
            self.ocr_available = False
        self.video_mode = False
        self.detection_active = False
    @property
    def model(self):
        """Lazy load YOLO model only when needed"""
        if self._model is None:
            print("Loading YOLO model...")
            self._model = YOLO(self.model_path)
            print("YOLO model loaded successfully")
        return self._model

    def clean_plate_text(self, text):
        """Clean up OCR results for license plates"""
        text = text.upper().strip()
        # Remove extra spaces and normalize
        text = re.sub(r'\s+', '', text)
        return text

    def validate_indian_plate_format(self, text):
        """Validate and format Indian license plate: 2 letters + 2 digits + 2 letters + 4 digits"""
        # Remove all spaces and special characters
        cleaned = re.sub(r'[^A-Z0-9,@]', '', text.upper())
        
        # Fix position-based OCR errors using Indian format knowledge
        if len(cleaned) == 10:
            corrected = list(cleaned)

            if corrected[2]=="B" and corrected[3]=="H":
                for i in [0,1]:
                    if corrected[i].isalpha():
                        # Common letter->digit mistakes
                        if corrected[i] == 'O': corrected[i] = '0'
                        elif corrected[i] == 'I': corrected[i] = '1'
                        elif corrected[i] == 'S': corrected[i] = '5'
                        elif corrected[i] == 'Z': corrected[i] = '2'
                        elif corrected[i] == 'B': corrected[i] = '8'
                        elif corrected[i] == 'G': corrected[i] = '6'
                        elif corrected[i] == 'J': corrected[i] = '3'
                        elif corrected[i] == 'E': corrected[i] = '3'
                for i in [0,1]:
                    if corrected[i].isalpha():
                        # Common letter->digit mistakes
                        if corrected[i] == 'O': corrected[i] = '0'
                        elif corrected[i] == 'I': corrected[i] = '1'
                        elif corrected[i] == 'S': corrected[i] = '5'
                        elif corrected[i] == 'Z': corrected[i] = '2'
                        elif corrected[i] == 'B': corrected[i] = '8'
                        elif corrected[i] == 'G': corrected[i] = '6'
                        elif corrected[i] == 'J': corrected[i] = '3'
                        elif corrected[i] == 'E': corrected[i] = '3'
                for i in [4,5,6, 7]:
                    if corrected[i].isalpha():
                        # Common letter->digit mistakes
                        if corrected[i] == 'O': corrected[i] = '0'
                        elif corrected[i] == 'I': corrected[i] = '1'
                        elif corrected[i] == 'S': corrected[i] = '5'
                        elif corrected[i] == 'Z': corrected[i] = '2'
                        elif corrected[i] == 'B': corrected[i] = '8'
                        elif corrected[i] == 'G': corrected[i] = '6'
                        elif corrected[i] == 'J': corrected[i] = '3'
                        elif corrected[i] == 'E': corrected[i] = '3'
                for i in [8, 9]:
                    if corrected[i].isdigit():
                        # Common digit->letter mistakes
                        if corrected[i] == '0': corrected[i] = 'O'
                        elif corrected[i] == '1': corrected[i] = 'I'
                        elif corrected[i] == '5': corrected[i] = 'S'
                        elif corrected[i] == '2': corrected[i] = 'Z'
                        elif corrected[i] == '8': corrected[i] = 'B'
                        elif corrected[i] == '6': corrected[i] = 'G'
           
                    
            else:
                # Positions 0,1 should be alphabets (A-Z)
                for i in [0, 1]:
                    if corrected[i].isdigit():
                        # Common digit->letter mistakes
                        if corrected[i] == '0': corrected[i] = 'O'
                        elif corrected[i] == '1': corrected[i] = 'I'
                        elif corrected[i] == '5': corrected[i] = 'S'
                        elif corrected[i] == '2': corrected[i] = 'Z'
                        elif corrected[i] == '8': corrected[i] = 'B'
                        elif corrected[i] == '6': corrected[i] = 'G'
                
                # Positions 2,3 should be digits (0-9)
                for i in [2, 3]:
                    if corrected[i].isalpha():
                        # Common letter->digit mistakes
                        if corrected[i] == 'O': corrected[i] = '0'
                        elif corrected[i] == 'I': corrected[i] = '1'
                        elif corrected[i] == 'S': corrected[i] = '5'
                        elif corrected[i] == 'Z': corrected[i] = '2'
                        elif corrected[i] == 'B': corrected[i] = '8'
                        elif corrected[i] == 'G': corrected[i] = '6'
                        elif corrected[i] == 'L': corrected[i] = '1'
                
                # Positions 4,5 should be alphabets (A-Z)
                for i in [4, 5]:
                    if corrected[i].isdigit() or corrected[i] == '@':
                        # Common digit->letter mistakes
                        if corrected[i] == '0': corrected[i] = 'D'
                        elif corrected[i] == '1': corrected[i] = 'I'
                        elif corrected[i] == '5': corrected[i] = 'S'
                        elif corrected[i] == '2': corrected[i] = 'Z'
                        elif corrected[i] == '8': corrected[i] = 'B'
                        elif corrected[i] == '6': corrected[i] = 'G'
                        elif corrected[i] == '3': corrected[i] = 'E'
                        elif corrected[i] == '4': corrected[i] = 'H'
                        elif corrected[i] == '@': corrected[i] = 'D'
                
                # Positions 6,7,8,9 should be digits (0-9)
                for i in [6, 7, 8, 9]:
                    if corrected[i].isalpha():
                        # Common letter->digit mistakes
                        if corrected[i] == 'O': corrected[i] = '0'
                        elif corrected[i] == 'I': corrected[i] = '1'
                        elif corrected[i] == 'S': corrected[i] = '5'
                        elif corrected[i] == 'Z': corrected[i] = '2'
                        elif corrected[i] == 'B': corrected[i] = '8'
                        elif corrected[i] == 'G': corrected[i] = '6'
                        elif corrected[i] == 'J': corrected[i] = '3'
                        elif corrected[i] == 'E': corrected[i] = '3'
                
                corrected_text = ''.join(corrected)
                return corrected_text, True
        

        # If doesn't match exact format, try to fix common issues
        elif len(cleaned) == 9:
                corrected = list(cleaned)

            
                for i in [0,1]:
                    if corrected[i].isdigit():
                        # Common digit->letter mistakes
                        if corrected[i] == '0': corrected[i] = 'O'
                        elif corrected[i] == '1': corrected[i] = 'I'
                        elif corrected[i] == '5': corrected[i] = 'S'
                        elif corrected[i] == '2': corrected[i] = 'Z'
                        elif corrected[i] == '8': corrected[i] = 'B'
                        elif corrected[i] == '6': corrected[i] = 'G'
                        elif corrected[i] == '4': corrected[i] = 'A'

                for i in [4]:
                    if corrected[i].isdigit():
                        # Common digit->letter mistakes
                        if corrected[i] == '0': corrected[i] = 'O'
                        elif corrected[i] == '1': corrected[i] = 'I'
                        elif corrected[i] == '5': corrected[i] = 'S'
                        elif corrected[i] == '2': corrected[i] = 'Z'
                        elif corrected[i] == '8': corrected[i] = 'B'
                        elif corrected[i] == '6': corrected[i] = 'G'
                        elif corrected[i] == '4': corrected[i] = 'A'

                for i in [2,3,5,6,7,8]:
                    if corrected[i].isalpha():
                        # Common letter->digit mistakes
                        if corrected[i] == 'O': corrected[i] = '0'
                        elif corrected[i] == 'I': corrected[i] = '1'
                        elif corrected[i] == 'S': corrected[i] = '5'
                        elif corrected[i] == 'Z': corrected[i] = '7'
                        elif corrected[i] == 'B': corrected[i] = '8'
                        elif corrected[i] == 'G': corrected[i] = '6'
                        elif corrected[i] == 'J': corrected[i] = '3'
                        elif corrected[i] == 'E': corrected[i] = '3'
                        
                
                corrected_text = ''.join(corrected)
                return corrected_text, True
        
    def preprocess_for_ocr(self, image):
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply some enhancement
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Denoise
            # Resize image if too small (OCR works better on larger images)
            height, width = gray.shape
            if height < 50 or width < 150:
                scale_factor = max(50 / height, 150 / width, 2.0)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            
            # Ensure the image is in the right format for EasyOCR
            # EasyOCR works better with PIL-compatible formats
            if PIL_AVAILABLE:
                # Convert to PIL Image and back to ensure compatibility
                pil_image = Image.fromarray(gray)
                processed_image = np.array(pil_image)
            else:
                processed_image = gray
            # Save preprocessed image for debugging
            cv2.imwrite(f'D:/AUTOTOLL/debug_preprocessed_de.jpg', processed_image)
            return processed_image
            
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            # Return original image if preprocessing fails
            return image

    def process_frame(self, frame, frame_idx=0, force_detection=False):
        """Process a single frame and return detected license plates"""        
        results = self.model(frame)
        plate_data = []
        crop_id = 0

        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Always draw YOLO detection box when detecting
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"YOLO: {confidence:.2f}", (x1, y1 - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    plate_roi = frame[y1:y2, x1:x2]

                    if plate_roi.size == 0:
                        continue

                    # Skip OCR processing for low YOLO confidence in video mode
                    if self.video_mode and confidence < 0.5 and not force_detection:
                        cv2.putText(frame, "OCR: Skipped (low YOLO conf)",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (128, 128, 128), 2)
                        crop_id += 1
                        continue

                    # Process OCR when detecting
                    if not self.ocr_available:
                        cv2.putText(frame, "OCR: Not Available",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (0, 0, 255), 2)
                        crop_id += 1
                        continue

                    try:
                        # Preprocess the ROI for better OCR
                        processed_roi = self.preprocess_for_ocr(plate_roi)
                        
                        # Ensure the image has proper dimensions
                        if processed_roi.shape[0] < 10 or processed_roi.shape[1] < 10:
                            cv2.putText(frame, "OCR: ROI too small",
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (128, 128, 128), 2)
                            crop_id += 1
                            continue
                        
                        # Try OCR with error handling
                        try:
                            ocr_results = self.reader.readtext(processed_roi)
                        except Exception as ocr_error:
                            # If preprocessing failed, try with original image
                            print(f"OCR with preprocessed image failed: {ocr_error}")
                            try:
                                ocr_results = self.reader.readtext(plate_roi)
                            except Exception as final_ocr_error:
                                print(f"OCR with original image also failed: {final_ocr_error}")
                                cv2.putText(frame, f"OCR: Failed ({str(final_ocr_error)[:15]})",
                                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, (0, 0, 255), 2)
                                crop_id += 1
                                continue
                        
                        if ocr_results:
                            # Sort text by x-coordinate to read left to right
                            sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])
                            
                            # Combine all text with reasonable confidence
                            combined_text = ""
                            total_conf = 0
                            valid_detections = 0
                            
                            for ocr_result in sorted_results:
                                text, conf = ocr_result[1], ocr_result[2]
                                if conf > 0.4:  # OCR confidence threshold
                                    combined_text += text.strip() + " "
                                    total_conf += conf
                                    valid_detections += 1
                            
                            if valid_detections > 0:
                                combined_text = combined_text.strip()
                                avg_conf = total_conf / valid_detections
                                
                                if avg_conf >= 0.3:
                                    # Clean the text
                                    cleaned_text = self.clean_plate_text(combined_text)
                                    
                                    # Apply format validation
                                    formatted_plate, is_valid = self.validate_indian_plate_format(cleaned_text)
                                    
                                    plate_info = {
                                        'plate': formatted_plate,
                                        'confidence': avg_conf,
                                        'yolo_confidence': confidence,
                                        'raw_text': combined_text,
                                        'bbox': (x1, y1, x2, y2),
                                        'is_valid_format': is_valid
                                    }
                                    plate_data.append(plate_info)
                                    
                                    if is_valid:
                                        # Draw OCR result box (green for valid)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                        cv2.putText(frame, f"OCR: {formatted_plate} ({avg_conf:.2f})",
                                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.7, (0, 255, 0), 2)
                                    else:
                                        # Draw OCR result box (yellow for non-standard)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                        cv2.putText(frame, f"OCR: {formatted_plate} ({avg_conf:.2f})",
                                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.7, (0, 255, 255), 2)
                                else:
                                    # Show OCR text even if confidence is low
                                    cv2.putText(frame, f"OCR: {combined_text} (low conf: {avg_conf:.2f})",
                                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                              0.6, (128, 128, 128), 2)
                            else:
                                # No valid OCR detections
                                cv2.putText(frame, "OCR: No text detected",
                                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6, (128, 128, 128), 2)
                        else:
                            # OCR found nothing
                            cv2.putText(frame, "OCR: Empty",
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (128, 128, 128), 2)
                        
                    except Exception as e:
                        error_msg = str(e)
                        # Truncate long error messages
                        if len(error_msg) > 20:
                            error_msg = error_msg[:17] + "..."
                        cv2.putText(frame, f"OCR: Error ({error_msg})",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 0, 255), 2)
                        print(f"Full OCR error: {e}")

                    crop_id += 1

        print("plate_data", plate_data)
        return frame, plate_data

    def set_video_mode(self, enabled=True):
        """Enable/disable video mode optimizations"""
        self.video_mode = enabled

    def set_detection_active(self, active=True):
        """Enable/disable detection processing"""
        self.detection_active = active

    def list_available_cameras(self):
        """List all available camera devices"""
        available_cameras = []
        
        for i in range(2):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Set a short timeout for reading
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
                cv2.destroyAllWindows()  # Clean up
        
        return available_cameras

