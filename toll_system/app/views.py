from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.core.cache import cache
from django.utils import timezone
from django.db.models import Sum, Count
from datetime import datetime, timedelta
from .models import Vehicle, TollRule, Transaction
# Fix the import - use relative import for the recognition module
try:
    from .recognition import PlateRecognition
except ImportError:
    # Fallback if there's a package structure issue
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from recognition import PlateRecognition


import cv2
import json
import threading
import time
from decimal import Decimal
import numpy as np
import tempfile
import os

# Global variables for camera system
camera = None
recognition_system = PlateRecognition()
camera_active = False
current_camera_index = 0
frame_lock = threading.Lock()
latest_frame = None

def index(request):
    """Enhanced home page with detailed statistics"""
    # Basic counts
    recent_transactions = Transaction.objects.order_by('-captured_at')[:10]
    total_vehicles = Vehicle.objects.count()
    total_transactions = Transaction.objects.count()
    
    # Today's statistics
    today = timezone.now().date()
    today_transactions = Transaction.objects.filter(captured_at__date=today)
    today_revenue = today_transactions.aggregate(Sum('toll_amount'))['toll_amount__sum'] or 0
    today_count = today_transactions.count()
    
    # Success rate calculation
    successful_transactions = Transaction.objects.filter(payment_status='success').count()
    success_rate = round((successful_transactions / total_transactions * 100) if total_transactions > 0 else 0, 1)
    
    # FASTag usage
    fastag_transactions = Transaction.objects.filter(payment_method='fastag').count()
    fastag_usage = round((fastag_transactions / total_transactions * 100) if total_transactions > 0 else 0, 1)
    
    # Detection accuracy (you can calculate this based on your needs)
    detection_accuracy = 90  # Default value, you can calculate this based on OCR confidence
    
    context = {
        'recent_transactions': recent_transactions,
        'total_vehicles': total_vehicles,
        'total_transactions': total_transactions,
        'today_revenue': today_revenue,
        'today_transactions': today_count,
        'success_rate': success_rate,
        'fastag_usage': fastag_usage,
        'detection_accuracy': detection_accuracy,
    }
    return render(request, 'index.html', context)

def camera_view(request):
    """Camera page for live detection"""
    cached_cameras = cache.get('available_cameras', [])
    
    context = {
        'available_cameras': cached_cameras,
        'camera_active': camera_active,
        'current_camera': current_camera_index if camera_active else None
    }
    return render(request, 'camera.html', context)

def manual_view(request):
    """Manual entry page"""
    if request.method == 'POST':
        plate = request.POST.get('plate', '').upper()
        if plate:
            result = process_plate(plate, 'manual')
            if result['status'] == 'success':
                messages.success(request, f'Transaction processed for {plate}')
            else:
                messages.error(request, result.get('message', 'Processing failed'))
    
    return render(request, 'manual.html')

def register_view(request):
    """Vehicle registration page"""
    if request.method == 'POST':
        plate = request.POST.get('plate', '').upper()
        owner_name = request.POST.get('owner_name', '')
        vehicle_type = request.POST.get('vehicle_type', '')
        fastag_id = request.POST.get('fastag_id', '')
        fastag_balance = request.POST.get('fastag_balance', 0)
        exemption = request.POST.get('exemption', '')
        
        try:
            vehicle, created = Vehicle.objects.get_or_create(
                plate=plate,
                defaults={
                    'owner_name': owner_name,
                    'vehicle_type': vehicle_type,
                    'fastag_id': fastag_id,
                    'exemption': exemption,
                    'fastag_balance': int(fastag_balance) if fastag_balance else 0
                }
            )
            
            if created:
                messages.success(request, f'Vehicle {plate} registered successfully!')
            else:
                messages.warning(request, f'Vehicle {plate} already exists!')
                
        except Exception as e:
            messages.error(request, f'Error registering vehicle: {str(e)}')
    
    return render(request, 'register.html')

def transactions_view(request):
    """Transaction history page"""
    transactions = Transaction.objects.order_by('-captured_at')
    return render(request, 'transactions.html', {'transactions': transactions})

@csrf_exempt
def list_cameras(request):
    """List available cameras with caching"""
    try:
        # Check cache first
        cameras = cache.get('available_cameras')
        if cameras is None:
            cameras = recognition_system.list_available_cameras()
            # Cache for 5 minutes
            cache.set('available_cameras', cameras, 100)
        
        return JsonResponse({
            'status': 'success',
            'cameras': cameras,
            'current_camera': current_camera_index if camera_active else None
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })
@csrf_exempt
def get_camera_status(request):
    """Get current camera status"""
    global camera_active, current_camera_index, recognition_system
    
    return JsonResponse({
        'camera_active': camera_active,
        'current_camera': current_camera_index,
        'detection_active': recognition_system.detection_active if recognition_system else False
    })

@csrf_exempt
def start_camera(request):
    """Start camera capture"""
    global camera, camera_active, current_camera_index
    
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body)
            camera_index = data.get('camera_index', 0)
            
            if camera_active:
                return JsonResponse({'status': 'error', 'message': 'Camera already active'})
            
            import cv2
            camera = cv2.VideoCapture(camera_index)
            
            if camera.isOpened():
                camera_active = True
                current_camera_index = camera_index
                return JsonResponse({'status': 'success', 'message': 'Camera started'})
            else:
                return JsonResponse({'status': 'error', 'message': 'Failed to open camera'})
                
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
def stop_camera(request):
    """Stop camera capture"""
    global camera, camera_active, current_camera_index
    
    if request.method == 'POST':
        try:
            if camera:
                camera.release()
                camera = None
            
            camera_active = False
            recognition_system.set_detection_active(False)
            
            return JsonResponse({'status': 'success', 'message': 'Camera stopped'})
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
def toggle_detection(request):
    """Toggle detection on/off"""
    if request.method == 'POST':
        try:
            if not camera_active:
                return JsonResponse({'status': 'error', 'message': 'Camera not active'})
            
            # Toggle detection state
            new_state = not recognition_system.detection_active
            recognition_system.set_detection_active(new_state)
            
            message = 'Detection started' if new_state else 'Detection stopped'
            
            return JsonResponse({
                'status': 'success', 
                'message': message,
                'detection_active': new_state
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def gen_frames():
    """Generate camera frames for video feed"""
    global camera, latest_frame
    
    while camera_active and camera:
        success, frame = camera.read()
        if not success:
            break
        
        # Store latest frame for capture (before processing)
        with frame_lock:
            latest_frame = frame.copy()
        
        # Process frame for display - only detect if detection is active
        processed_frame, plate_data = recognition_system.process_frame(frame, force_detection=False)
        
        # Add status overlay to show detection state
        if recognition_system.detection_active:
            cv2.putText(processed_frame, "DETECTION: ACTIVE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(processed_frame, "DETECTION: INACTIVE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@csrf_exempt
def capture_frame(request):
    """Capture and process current frame"""
    global latest_frame
    
    if request.method == 'POST':
        try:
            if not camera_active or latest_frame is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Camera not active or no frame available'
                })
            
            with frame_lock:
                frame_copy = latest_frame.copy()
            
            # Process frame with detection FORCED ON for capture
            processed_frame, plate_data = recognition_system.process_frame(frame_copy, force_detection=True)
            
            results = []
            if plate_data:
                for plate_info in plate_data:
                    result = process_plate(
                        plate_info['plate'], 
                        'camera_capture', 
                        plate_info
                    )
                    results.append(result)
            
            if results:
                return JsonResponse({
                    'status': 'success',
                    'message': f'Captured and processed {len(results)} plates',
                    'results': results
                })
            else:
                return JsonResponse({
                    'status': 'warning',
                    'message': 'No license plates detected in captured frame'
                })
                
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def video_feed(request):
    """Video feed endpoint"""
    if not camera_active:
        # Return a placeholder image
        placeholder = create_placeholder_frame()
        ret, buffer = cv2.imencode('.jpg', placeholder)
        return StreamingHttpResponse(
            iter([buffer.tobytes()]),
            content_type='image/jpeg'
        )
    
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def create_placeholder_frame():
    """Create a placeholder frame when camera is off"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, 'Camera Offline', (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def process_plate(plate, source, detection=None):
    """Process license plate and create transaction"""
    try:
        # Look up vehicle
        try:
            vehicle = Vehicle.objects.get(plate=plate)
        except Vehicle.DoesNotExist:
            vehicle = None
        
        # Check if vehicle is blacklisted
        if vehicle and vehicle.blacklist:
            return {
                'status': 'blocked',
                'message': 'Vehicle is blacklisted',
                'plate': plate
            }
        
        # Check exemption
        if vehicle and vehicle.exemption:
            toll_amount = Decimal('0.00')
            payment_status = 'exempted'
            payment_method = 'exemption'
        else:
            # Calculate toll based on vehicle type
            vehicle_type = vehicle.vehicle_type
            
            try:
                toll_rule = TollRule.objects.get(vehicle_type=vehicle_type)
                toll_amount = toll_rule.base_amount
            except TollRule.DoesNotExist:
                # Default toll amount
                toll_amount = Decimal('50.00')
            
            # Process payment
            if vehicle and vehicle.fastag_balance and vehicle.fastag_balance >= toll_amount:
                # Deduct from FASTag
                vehicle.fastag_balance -= int(toll_amount)
                vehicle.save()
                payment_status = 'success'
                payment_method = 'fastag'
            else:
                payment_status = 'pending'
                payment_method = 'cash'
        
        # Create transaction
        transaction = Transaction.objects.create(
            plate=plate,
            toll_amount=toll_amount,
            payment_method=payment_method,
            payment_status=payment_status,
            location='Main Toll Plaza',
            cam_id='CAM_001',
            raw_ocr_text=detection['raw_text'] if detection else plate,
            ocr_confidence=detection['confidence'] if detection else None,
            notes=f'Processed via {source}'
        )
        
        return {
            'status': 'success',
            'plate': plate,
            'toll_amount': str(toll_amount),
            'payment_status': payment_status,
            'payment_method': payment_method,
            'vehicle_type': vehicle_type,
            'owner': vehicle.owner_name if vehicle else 'Unknown',
            'transaction_id': transaction.id,
            'exemption': vehicle.exemption if vehicle else False
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'plate': plate
        }

@csrf_exempt
def get_live_detections(request):
    """Get current live detection results"""
    global latest_frame
    
    if request.method == 'GET':
        try:
            if not camera_active or latest_frame is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Camera not active or no frame available',
                    'detections': []
                })
            
            if not recognition_system.detection_active:
                return JsonResponse({
                    'status': 'inactive',
                    'message': 'Detection is not active',
                    'detections': []
                })
            
            with frame_lock:
                frame_copy = latest_frame.copy()
            
            # Process frame for detection data only (no visual processing)
            _, plate_data = recognition_system.process_frame(frame_copy, force_detection=True)
            
            detection_results = []
            if plate_data:
                for plate_info in plate_data:
                    detection_results.append({
                        'plate': plate_info['plate'],
                        'confidence': round(plate_info['confidence'], 3),
                        'yolo_confidence': round(plate_info['yolo_confidence'], 3),
                        'raw_text': plate_info['raw_text'],
                        'is_valid': plate_info['is_valid_format'],
                        'timestamp': time.time()
                    })
            
            return JsonResponse({
                'status': 'success',
                'detections': detection_results,
                'detection_active': recognition_system.detection_active
            })
                
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e),
                'detections': []
            })
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
def process_client_frame(request):
    """Process frames sent from client-side camera"""
    if request.method == 'POST' and request.FILES.get('frame'):
        try:
            frame_file = request.FILES['frame']
            single_capture = request.POST.get('single_capture', '0') == '1'
            
            # Create a temporary file to save the uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                for chunk in frame_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            # Read the image with OpenCV
            frame = cv2.imread(tmp_path)
            
            # Clean up the temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            if frame is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to read uploaded image'
                })
            
            # Process the frame with our recognition system
            processed_frame, plate_data = recognition_system.process_frame(frame, force_detection=True)
            
            if single_capture:
                # For manual capture, process plates and return full results
                results = []
                if plate_data:
                    for plate_info in plate_data:
                        result = process_plate(
                            plate_info['plate'], 
                            'client_capture', 
                            plate_info
                        )
                        results.append(result)
                
                return JsonResponse({
                    'status': 'success',
                    'message': f'Processed {len(results)} plates' if results else 'No plates detected',
                    'results': results
                })
            else:
                # For continuous detection, just return detection data
                detection_results = []
                if plate_data:
                    for plate_info in plate_data:
                        detection_results.append({
                            'plate': plate_info['plate'],
                            'confidence': round(plate_info['confidence'], 3),
                            'yolo_confidence': round(plate_info['yolo_confidence'], 3),
                            'raw_text': plate_info['raw_text'],
                            'is_valid': plate_info['is_valid_format'],
                            'timestamp': time.time()
                        })
                
                return JsonResponse({
                    'status': 'success',
                    'detections': detection_results
                })
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    })
