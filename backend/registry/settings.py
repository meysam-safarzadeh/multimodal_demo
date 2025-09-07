import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # loads backend/.env

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
SECRET_KEY = "dev-secret-key-change-me"
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "real-concrete-caribou.ngrok-free.app"]
CALLBACK_SECRET = os.getenv("CALLBACK_SECRET", "change-me")
CALLBACK_TTL = int(os.getenv("CALLBACK_TTL", "900"))  # 15 min


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
    "NAME": os.getenv("POSTGRES_NAME", "multimodal_demo"),
    "USER": os.getenv("POSTGRES_USER", "postgres"),
    "PASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "HOST": os.getenv("POSTGRES_HOST", "localhost"),
    "PORT": os.getenv("POSTGRES_PORT", "5432"),
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

# ECS config (available as settings.ECS)
ECS = {
    "AWS_REGION": os.getenv("AWS_REGION", "eu-central-1"),
    "CLUSTER": os.getenv("ECS_CLUSTER", ""),
    "TASK_DEFINITION": os.getenv("ECS_TASK_DEFINITION", ""),
    "CONTAINER_NAME": os.getenv("ECS_CONTAINER_NAME", "trainer"),
    "SUBNETS": [s for s in os.getenv("ECS_SUBNETS", "").split(",") if s],
    "SECURITY_GROUPS": [s for s in os.getenv("ECS_SECURITY_GROUPS", "").split(",") if s],
    "ASSIGN_PUBLIC_IP": os.getenv("ECS_ASSIGN_PUBLIC_IP", "ENABLED"),
    "PLATFORM_VERSION": os.getenv("ECS_PLATFORM_VERSION", "LATEST"),
}
S3_BUCKET = os.getenv("S3_BUCKET", "demo-bucket")
