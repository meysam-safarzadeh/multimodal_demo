from rest_framework import serializers
from .models import Dataset, TrainingJob, Artifact, TrainingMetrics
from django.core.validators import URLValidator

# Custom URL validator that accepts s3:// in addition to http/https
class S3URLValidator(URLValidator):
    schemes = ["http", "https", "s3"]

class S3URLField(serializers.CharField):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validators.append(S3URLValidator())

class DatasetSerializer(serializers.ModelSerializer):
    csv_s3_uri = S3URLField()
    data_s3_uri = S3URLField()

    class Meta:
        model = Dataset
        fields = "__all__"

class TrainingMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingMetrics
        fields = ["accuracy", "loss", "val_accuracy", "val_loss", "logs", "created_at", "updated_at"]


class TrainingJobSerializer(serializers.ModelSerializer):
    metrics = TrainingMetricsSerializer(read_only=True)

    class Meta:
        model = TrainingJob
        fields = "__all__"

# class MetricSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Metric
#         fields = "__all__"

class ArtifactSerializer(serializers.ModelSerializer):
    class Meta:
        model = Artifact
        fields = "__all__"