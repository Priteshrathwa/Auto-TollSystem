from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('camera/', views.camera_view, name='camera'),
    path('manual/', views.manual_view, name='manual'),
    path('register/', views.register_view, name='register'),
    path('transactions/', views.transactions_view, name='transactions'),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('list_cameras/', views.list_cameras, name='list_cameras'),
    path('get_camera_status/', views.get_camera_status, name='get_camera_status'),
    path('start_camera/', views.start_camera, name='start_camera'),
    path('stop_camera/', views.stop_camera, name='stop_camera'),
    path('toggle_detection/', views.toggle_detection, name='toggle_detection'),
    path('capture_frame/', views.capture_frame, name='capture_frame'),
    path('live_detections/', views.get_live_detections, name='live_detections'),
    path('process_client_frame/', views.process_client_frame, name='process_client_frame'),
]
