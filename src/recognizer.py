# src/recognizer.py
"""License plate recognition using OpenCV and EasyOCR."""
import cv2
import easyocr
import numpy as np
import logging
import re
import os
from src.utils import load_image, PlateNotDetectedError

class LicensePlateRecognizer:
    """Class to detect and read license plates from images or videos."""
    
    def __init__(self, ocr_lang='en'):
        """Initialize the recognizer with OCR language."""
        try:
            self.reader = easyocr.Reader([ocr_lang])
            logging.info(f"Initialized EasyOCR with language: {ocr_lang}")
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def _preprocess(self, image, target_width=800):
        """Enhance contrast, denoise, gamma correct and resize for better detection."""
        # Convert to gray and apply CLAHE
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # Gamma correction (slightly brighten)
        gamma = 1.1
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        gray = cv2.LUT(gray, table)
        # Resize to target width keeping aspect ratio to normalize detection scale
        h, w = gray.shape
        if w < target_width:
            scale = target_width / w
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        elif w > target_width * 2:
            scale = target_width / w
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        # Also provide an RGB version for EasyOCR (expects RGB)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return gray, rgb

    def _four_point_transform(self, image, pts):
        """Perform a perspective transform to obtain a top-down view."""
        rect = np.array(pts, dtype="float32")
        # order points: tl, tr, br, bl
        s = rect.sum(axis=1)
        diff = np.diff(rect, axis=1)
        tl = rect[np.argmin(s)]
        br = rect[np.argmax(s)]
        tr = rect[np.argmin(diff)]
        bl = rect[np.argmax(diff)]
        rect_ord = np.array([tl, tr, br, bl], dtype="float32")
        (tl, tr, br, bl) = rect_ord
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect_ord, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def detect_plate(self, image):
        """Detect license plate in an image using OCR-first then contour fallback."""
        try:
            gray, rgb = self._preprocess(image)
            # Strategy A: EasyOCR detection boxes -> merge suitable boxes
            try:
                ocr_results = self.reader.readtext(rgb, detail=1)
                candidates = []
                for bbox, text, conf in ocr_results:
                    # bbox is list of 4 points; compute bounding rect
                    pts = np.array(bbox, dtype=np.float32)
                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]
                    x1, y1 = x_coords.min(), y_coords.min()
                    x2, y2 = x_coords.max(), y_coords.max()
                    w = x2 - x1
                    h = y2 - y1 if (y2 - y1) > 0 else 1
                    ar = w / h
                    # heuristic: plate-like aspect ratio and some confidence
                    if 2.0 <= ar <= 8.0 and conf > 0.3:
                        candidates.append(((int(x1), int(y1), int(x2), int(y2)), conf))
                if candidates:
                    # Merge all candidate boxes into a single ROI
                    xs = [c[0][0] for c in candidates] + [c[0][2] for c in candidates]
                    ys = [c[0][1] for c in candidates] + [c[0][3] for c in candidates]
                    x1, y1 = max(min(xs) - 8, 0), max(min(ys) - 8, 0)
                    x2, y2 = min(max(xs) + 8, rgb.shape[1]-1), min(max(ys) + 8, rgb.shape[0]-1)
                    roi = gray[y1:y2+1, x1:x2+1]
                    logging.info("Plate region found using EasyOCR detection boxes.")
                    return roi
            except Exception as e:
                logging.debug(f"EasyOCR detection step failed or returned no boxes: {e}")

            # Strategy B: contour + minAreaRect with morphological ops
            try:
                # morphological closing to connect characters
                kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
                morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kern)
                # edge detection on morph result
                edged = cv2.Canny(morph, 50, 200)
                contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 1000:
                        continue
                    rect = cv2.minAreaRect(cnt)
                    (w, h) = rect[1]
                    if w == 0 or h == 0:
                        continue
                    ar = max(w, h) / (min(w, h) + 1e-6)
                    # plate-like aspect ratio and reasonable size
                    if 2.0 <= ar <= 8.0 and max(w, h) > 40:
                        box = cv2.boxPoints(rect)
                        warped = self._four_point_transform(gray, box)
                        # ensure width > height
                        if warped.shape[1] < warped.shape[0]:
                            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                        candidates.append((warped, area))
                if candidates:
                    # pick largest area candidate
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    logging.info("Plate region found using contour/minAreaRect method.")
                    roi = candidates[0][0]
                    # upscale small plate regions to improve OCR
                    if roi.shape[1] < 200:
                        scale = int(200 / roi.shape[1]) + 1
                        roi = cv2.resize(roi, (roi.shape[1]*scale, roi.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
                    return roi
            except Exception as e:
                logging.debug(f"Contour-based detection failed: {e}")

            # Strategy C: fallback to full image (preprocessed)
            logging.warning("No plate region found; falling back to full preprocessed image for OCR.")
            return gray

        except Exception as e:
            logging.error(f"Error detecting license plate: {e}")
            raise

    def read_text(self, cropped_image):
        """Read text from cropped license plate using EasyOCR with multi-scale attempts."""
        try:
            candidates = []
            # Try multiple scales (original, upscaled x2, sharpened)
            scales = [1.0, 1.5, 2.0]
            for s in scales:
                if s != 1.0:
                    img = cv2.resize(cropped_image, (int(cropped_image.shape[1]*s), int(cropped_image.shape[0]*s)),
                                     interpolation=cv2.INTER_CUBIC)
                else:
                    img = cropped_image
                # convert to RGB as EasyOCR expects
                if len(img.shape) == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    results = self.reader.readtext(img_rgb, detail=1)
                    for bbox, text, conf in results:
                        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                        # require at least 3 chars to avoid noise
                        if len(cleaned) >= 3:
                            candidates.append((cleaned, conf))
                except Exception:
                    continue
            if not candidates:
                raise PlateNotDetectedError("No text detected on license plate.")
            # pick highest confidence result
            candidates.sort(key=lambda x: x[1], reverse=True)
            text = candidates[0][0]
            confidence = candidates[0][1]
            logging.info(f"Detected text: {text} (Confidence: {confidence:.2f})")
            return text
        except Exception as e:
            logging.error(f"Error reading license plate text: {e}")
            raise
    
    def process_image(self, image_path):
        """Process an image to detect and read license plate."""
        try:
            image = load_image(image_path)
            if image is None:
                raise PlateNotDetectedError(f"Failed to load image: {image_path}")
            
            cropped_image = self.detect_plate(image)
            plate_text = self.read_text(cropped_image)
            return plate_text
        except PlateNotDetectedError as e:
            logging.error(f"Plate processing failed: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error processing image {image_path}: {e}")
            raise
    
    def process_video(self, video_path):
        """Process a video stream to detect plates frame-by-frame."""
        logging.warning("Video processing not implemented yet.")
        pass  # To be implemented as a future enhancement

    def detect_vehicle_type(self, image, model_paths=None):
        """Detect vehicle type (car, truck, bus, motorbike, unknown).
        
        Strategy:
        - If model_paths provided and model files exist, run a lightweight DNN (e.g., MobileNet-SSD).
        - Otherwise, fall back to a heuristic: find large external contour(s), compute bounding box aspect ratio and area,
          and classify by simple thresholds (works reasonably for cropped/zoomed/bad-angle images).
        
        model_paths can be a dict: {'prototxt': 'path', 'model': 'path'}.
        """
        try:
            # Normalize size for detection heuristics
            h0, w0 = image.shape[:2]
            target_w = 800
            scale = target_w / w0 if w0 and w0 < target_w else 1.0
            if scale != 1.0:
                img = cv2.resize(image, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_CUBIC)
            else:
                img = image.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Option A: DNN if model files provided
            try:
                if model_paths and os.path.exists(model_paths.get('prototxt', '')) and os.path.exists(model_paths.get('model', '')):
                    net = cv2.dnn.readNetFromCaffe(model_paths['prototxt'], model_paths['model'])
                    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
                    net.setInput(blob)
                    detections = net.forward()
                    # Mapping (common MobileNet-SSD class ids): 7=car, 14=motorbike, 6=bus, 8=truck
                    label_map = {7: 'car', 14: 'motorbike', 6: 'bus', 8: 'truck'}
                    counts = {}
                    (h, w) = img.shape[:2]
                    for i in range(detections.shape[2]):
                        conf = detections[0, 0, i, 2]
                        if conf < 0.4:
                            continue
                        idx = int(detections[0, 0, i, 1])
                        label = label_map.get(idx)
                        if label:
                            counts[label] = counts.get(label, 0) + 1
                    if counts:
                        # pick most frequent detection
                        vehicle_type = max(counts.items(), key=lambda x: x[1])[0]
                        logging.info(f"Vehicle type (DNN) detected: {vehicle_type}")
                        return vehicle_type
            except Exception as e:
                logging.debug(f"DNN-based vehicle detection failed or models missing: {e}")

            # Option B: Heuristic contour-based detection
            try:
                # Preprocess to connect vehicle silhouette / large shapes
                blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern, iterations=2)
                contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    logging.debug("No contours found for vehicle heuristic.")
                    return "unknown"
                # pick the largest contour by area
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 2000:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = w / (h + 1e-6)
                    # Adjust thresholds depending on scale (image width)
                    abs_w = w
                    # Heuristics:
                    # - motorbike: small width and low area, aspect ~ 1.0-2.0
                    # - car: aspect roughly 1.4-3.0
                    # - truck/bus: aspect larger, and absolute width bigger
                    if abs_w < 150 and area < 10000:
                        logging.info("Vehicle heuristic: motorbike")
                        return "motorbike"
                    if 1.2 <= aspect <= 3.0:
                        logging.info("Vehicle heuristic: car")
                        return "car"
                    if aspect > 3.0 or abs_w > 400:
                        logging.info("Vehicle heuristic: truck")
                        return "truck"
                    # fallback to checking proportion of image occupied
                    prop = area / float(img.shape[0] * img.shape[1])
                    if prop > 0.25:
                        logging.info("Vehicle heuristic: bus/truck (large proportion)")
                        return "truck"
                logging.info("Vehicle heuristic: unknown (no decisive contour)")
                return "unknown"
            except Exception as e:
                logging.debug(f"Contour heuristic failed: {e}")
                return "unknown"

        except Exception as e:
            logging.error(f"Error detecting vehicle type: {e}")
            return "unknown"

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python src/recognizer.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    recognizer = LicensePlateRecognizer()
    try:
        plate_text = recognizer.process_image(image_path)
        print(f"Detected License Plate: {plate_text}")
    except Exception as e:
        print(f"Error: {e}")
