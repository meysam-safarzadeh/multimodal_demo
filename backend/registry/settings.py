import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "dev-secret-key-change-me"
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

INSTALLED_APPS = [
    # Django core apps required for admin
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Your app with the models
    "registry",  # or "registry.apps.RegistryConfig" if you have apps.py
    "rest_framework",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",            # REQUIRED
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",        # REQUIRED
    "django.contrib.messages.middleware.MessageMiddleware",           # REQUIRED
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "registry.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",  # REQUIRED
        "DIRS": [],
        "APP_DIRS": True,                                              # REQUIRED for admin templates
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

WSGI_APPLICATION = "registry.wsgi.application"

# --- Postgres DB config ---
DATABASES = {
  "default": {
    "ENGINE": "django.db.backends.postgresql",
    "NAME": "my_db",
    "USER": "postgres",
    "PASSWORD": "QWER4321",
    "HOST": "localhost",
    "PORT": "5432",
  }
}

# Static files
STATIC_URL = "static/"

# Use BigAutoField to silence the warnings you saw
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Locale
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
