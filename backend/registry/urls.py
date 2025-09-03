from django.urls import path, include
from django.contrib import admin
from rest_framework.routers import DefaultRouter
from registry.views import (
    DatasetViewSet, TrainingJobViewSet, ArtifactViewSet
)

router = DefaultRouter()
router.register(r"datasets", DatasetViewSet)
router.register(r"training_jobs", TrainingJobViewSet)
router.register(r"artifacts", ArtifactViewSet)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include(router.urls)),
    path("", include(router.urls)),
]