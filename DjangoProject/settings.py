# DjangoProject/settings.py
from pathlib import Path
from datetime import timedelta
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJ_SECRET_KEY", "dev-secret-key-change-me")
DEBUG = True

ALLOWED_HOSTS = ["*",]  # dev only; lock down in prod

NNUNET_RESULTS_DIR = str(Path(BASE_DIR) / "HeartGuardBE" / "nnUNet_results")
NNUNET_CHECKPOINT = "checkpoint_best.pth"
NNUNET_CFG = "2d"

NNUNET_DATASET_3X4PLUS1 = "503"   # your Dataset503...
NNUNET_DATASET_3X4 = "504"        # your Dataset504...
NNUNET_FOLDS = ["0", "1", "2", "3", "4"]


INSTALLED_APPS = [
    # Jazzmin admin theme (must be before django.contrib.admin)
    "jazzmin",

    # third-party
    "rest_framework",
    # "rest_framework.authtoken",
    "corsheaders",

    # django
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # local
    "HeartGuardBE",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",  # must be high
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "DjangoProject.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "DjangoProject.wsgi.application"

# DATABASES: keep your existing config or env vars
# Example SQLite for dev:

# DATABASE (SQLite for local development) 
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}



AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ────────── DRF + JWT ──────────
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        "rest_framework.permissions.IsAuthenticated",
    ),
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
}

from rest_framework.settings import api_settings  # noqa

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=30),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "AUTH_HEADER_TYPES": ("Bearer", "JWT"),
}

# ────────── CORS (dev) ──────────
CORS_ALLOW_ALL_ORIGINS = True  # dev only
CORS_ALLOW_CREDENTIALS = True
CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://10.0.2.2:8000",
    "http://192.168.0.0:8000",  # adjust/LAN if needed
]

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
