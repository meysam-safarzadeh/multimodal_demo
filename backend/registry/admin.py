from django.contrib import admin
from .models import Dataset, TrainingArtifacts, TrainingJob, TrainingMetrics

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "csv_s3_uri", "data_s3_uri", "created_at")
    search_fields = ("name",)

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "job_name", "status", "started_at", "ended_at")
    list_filter = ("status",)

@admin.register(TrainingMetrics)
class TrainingMetricsAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "accuracy", "loss", "val_accuracy", "val_loss", "logs", "created_at", "updated_at")

@admin.register(TrainingArtifacts)
class TrainingArtifactsAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "s3_uri", "created_at")
