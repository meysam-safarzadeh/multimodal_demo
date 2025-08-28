from django.contrib import admin
from .models import Dataset, DetectionResult, TrainingJob, Metric, Artifact

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "csv_s3_uri", "data_s3_uri", "created_at")
    search_fields = ("name",)

@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "n_rows", "n_columns", "created_at")

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "job_name", "status", "started_at", "ended_at")
    list_filter = ("status",)

@admin.register(Metric)
class MetricAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "key", "value_num", "value_text")

@admin.register(Artifact)
class ArtifactAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "kind", "s3_uri", "created_at")
