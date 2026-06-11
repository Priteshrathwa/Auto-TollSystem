import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

import os

SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'unsafe-default-key')


DEBUG = True

ALLOWED_HOSTS = ['*']

# Required Python packages for this project:
# pip install django mysql-connector-python opencv-python easyocr ultralytics pillow numpy

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'toll_system.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'toll_system.wsgi.application'


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'tollsystem',
        'USER': 'root',
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
# =======================
# Custom Toll Config
# =======================

# YOLO Model
YOLO_MODEL_PATH = r"D:\AUTOTOLL\yolo model\best.pt"




# Camera
DEFAULT_CAMERA_INDEX = 0
CAMERA_RESOLUTION_WIDTH = 960
CAMERA_RESOLUTION_HEIGHT = 540
CAMERA_FPS = 30



AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "app" / "static",
]

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
