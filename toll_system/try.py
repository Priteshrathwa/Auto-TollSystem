import cv2
import easyocr
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("D:/AUTOTOLL/yolo model/best.pt")  # replace with your weights
reader = easyocr.Reader(['en'])

# Config
OCR_CONFIDENCE_THRESHOLD = 0.1   # Lower threshold for better detection
FRAME_SKIP = 1  # Process every frame (no skipping)
DEBUG_SAVE = False  # Disable debug saving for video performance
DEBUG_DIR = "debug_plates"
VIDEO_MODE = False  # Flag to optimize for video processing

if DEBUG_SAVE and not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

def clean_plate_text(text):
    """Clean up OCR results for license plates"""
    import re
    # Remove common OCR misreadings and clean up
    text = text.upper().strip()
    # Remove non-alphanumeric characters except spaces
    
    # Remove extra spaces and normalize
    text = re.sub(r'\s+', '', text)
    # Only keep general OCR mistake corrections (no specific plate cases)
    return text

def validate_indian_plate_format(text):
    """Validate and format Indian license plate: 2 letters + 2 digits + 2 letters + 4 digits"""
    import re
    
    # Remove all spaces and special characters
    #cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    cleaned = re.sub(r'[^A-Z0-9,@]', '', text.upper())
    print(f"🔧 Cleaned text: '{cleaned}' (length: {len(cleaned)})")
    
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
                    elif corrected[i] == 'Z': corrected[i] = '2'
                    elif corrected[i] == 'B': corrected[i] = '8'
                    elif corrected[i] == 'G': corrected[i] = '6'
                    elif corrected[i] == 'J': corrected[i] = '3'
                    elif corrected[i] == 'E': corrected[i] = '3'
                    
            
            corrected_text = ''.join(corrected)
            return corrected_text, True

def process_frame(frame, frame_idx=0):
    results = model(frame)
    plate_numbers = []
    crop_id = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Always draw YOLO detection box first
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for YOLO detection
            cv2.putText(frame, f"YOLO: {confidence:.2f}", (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            plate_roi = frame[y1:y2, x1:x2]

            if plate_roi.size == 0:
                continue

            # Save debug image only if enabled and not in video mode
            if DEBUG_SAVE and not VIDEO_MODE:
                debug_path = os.path.join(DEBUG_DIR, f"frame{frame_idx}_plate{crop_id}.jpg")
                cv2.imwrite(debug_path, plate_roi)

            # Skip OCR processing for low YOLO confidence in video mode
            if VIDEO_MODE and confidence < 0.5:
                cv2.putText(frame, "OCR: Skipped (low YOLO conf)",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (128, 128, 128), 2)
                crop_id += 1
                continue

            # Simple OCR processing - no heavy preprocessing
            try:
                ocr_results = reader.readtext(plate_roi)
                
                # Reduce console output in video mode
                if not VIDEO_MODE:
                    print(f"🔎 Frame {frame_idx}, Crop {crop_id}: YOLO conf={confidence:.3f}, OCR: {ocr_results}")
                
                if ocr_results:
                    # Sort text by x-coordinate to read left to right
                    sorted_results = sorted(ocr_results, key=lambda x: x[0][0][0])
                    
                    # Combine all text with reasonable confidence
                    combined_text = ""
                    total_conf = 0
                    valid_detections = 0
                    
                    for result in sorted_results:
                        text, conf = result[1], result[2]
                        if conf > OCR_CONFIDENCE_THRESHOLD:
                            combined_text += text.strip() + " "
                            total_conf += conf
                            valid_detections += 1
                    
                    if valid_detections > 0:
                        combined_text = combined_text.strip()
                        avg_conf = total_conf / valid_detections
                        
                        print(f"🏆 Combined text: '{combined_text}' (conf: {avg_conf:.3f})")
                        
                        if avg_conf >= OCR_CONFIDENCE_THRESHOLD:
                            # Clean the text
                            cleaned_text = clean_plate_text(combined_text)
                            
                            # Apply format validation
                            formatted_plate, is_valid = validate_indian_plate_format(cleaned_text)
                            
                            if is_valid:
                                print(f"✅ Valid Indian format: {formatted_plate}")
                                plate_numbers.append(formatted_plate)
                                
                                # Draw OCR result box (green for valid)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(frame, f"OCR: {formatted_plate} ({avg_conf:.2f})",
                                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.7, (0, 255, 0), 2)
                            else:
                                print(f"⚠️ Non-standard format: {formatted_plate}")
                                plate_numbers.append(formatted_plate)
                                
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
                print(f"❌ OCR error: {e}")
                cv2.putText(frame, f"OCR: Error ({str(e)[:20]})",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.6, (0, 0, 255), 2)

            crop_id += 1

    return frame, plate_numbers
def run_on_camera(camera_index=0):
    global VIDEO_MODE
    VIDEO_MODE = True  # Enable video optimizations
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    detected_plates = set()  # Track unique plates
    
    print("📷 Camera started. Press 'q' to quit, Enter for snapshot, 'c' to clear detected plates")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not accessible")
            break

        frame_count += 1
        
        # HUD overlays
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Unique Plates: {len(detected_plates)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press Enter to detect this frame (snapshot)", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("License Plate Detection (Camera)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # quit
            break
        elif key == ord('c'):  # clear
            detected_plates.clear()
            print("🧹 Cleared detected plates list")
        elif key in (13, 10):  # Enter key -> process only current frame
            snapshot = frame.copy()
            prev_video_mode = VIDEO_MODE
            VIDEO_MODE = False  # disable optimizations for snapshot
            processed_frame, plates = process_frame(snapshot, frame_idx=frame_count)
            VIDEO_MODE = prev_video_mode
            if plates:
                for plate in plates:
                    if plate not in detected_plates:
                        detected_plates.add(plate)
                        print(f"✅ NEW Plate: {plate}")
            else:
                print(f"📋 Frame {frame_count}: No plate detected")
            
            window_name = f"Detection Snapshot - Frame {frame_count}"
            cv2.imshow(window_name, processed_frame)
            cv2.waitKey(0)  # wait for user keypress before closing
            cv2.destroyWindow(window_name)

    cap.release()
    cv2.destroyAllWindows()
    VIDEO_MODE = False  # Reset



def run_on_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Error: Could not load image")
        return
    frame, plates = process_frame(frame, frame_idx=0)
    if plates:
        print("✅ Detected Plates:", plates)
    else:
        print("⚠️ No plate detected!")
    scale = 0.5  # 50%
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.namedWindow("Result - Image", cv2.WINDOW_NORMAL)  # create resizable window
    cv2.imshow("Result - Image", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_video(video_path):
    global VIDEO_MODE, tracked_cars
    VIDEO_MODE = True  # Enable video optimizations
    tracked_cars = {}  # Reset tracking
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return
    
    # Get video properties for optimization
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video: {fps:.1f} FPS, {total_frames} frames")

    # Playback pacing: derive delay from FPS (fallback to ~30 FPS)
    base_delay_ms = int(1000 / fps) if fps and fps > 0 else 33
    speed_factor = 1.0  # <1.0 faster, >1.0 slower

    frame_count = 0
    print("▶️ Press Enter to detect only the current frame. 'q' quit, 'f' faster, 's' slower.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hint overlay for snapshot detection
        cv2.putText(frame, "Press Enter to detect this frame (snapshot)", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        frame_count += 1
        
        # HUD overlays
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracked Cars: {len(tracked_cars)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Delay: {max(1, int(base_delay_ms * speed_factor))}ms (f/s to adjust)", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("License Plate Detection", frame)
        
        # Controls with FPS-based delay
        delay = max(1, int(base_delay_ms * speed_factor))
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key in (13, 10):  # Enter key -> process only the current frame
            snapshot = frame.copy()
            # Temporarily disable video shortcuts for better OCR on snapshot
            prev_video_mode = VIDEO_MODE
            VIDEO_MODE = False
            processed_frame, plates = process_frame(snapshot, frame_idx=frame_count)
            VIDEO_MODE = prev_video_mode
            if plates:
                print(f"📋 Frame {frame_count} plates: {', '.join(plates)}")
            else:
                print(f"📋 Frame {frame_count}: No plate detected")
            window_name = f"Detection Snapshot - Frame {frame_count}"
            cv2.imshow(window_name, processed_frame)
            # Pause to let user review snapshot; close on any key
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
        elif key == ord('f'):  # faster
            speed_factor = max(0.2, speed_factor * 0.7)
        elif key == ord('s'):  # slower
            speed_factor = min(5.0, speed_factor * 1.3)
    
    print(f"🏁 Video processing complete. Total tracked cars: {len(tracked_cars)}")
    print(f"🚗 Final tracked cars with best plates:")
    for car_id, data in tracked_cars.items():
        print(f"  Car-{car_id}: {data['plate']} (confidence: {data['confidence']:.3f})")
    
    cap.release()
    cv2.destroyAllWindows()
    VIDEO_MODE = False  # Reset video mode

# -------------------------------
# 🔹 Choose input file
# -------------------------------
#file_path = "3.png"   # or "traffic.mp4"
# file_path = "car.jpg"      # for image
file_path = ""  # for video
run_on_camera(0)
           # for live camera
 
if file_path.lower().endswith((".jpg", ".png", ".jpeg")):
    run_on_image(file_path)
elif file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
    run_on_video(file_path)
else:
    print("❌ Unsupported file format")

