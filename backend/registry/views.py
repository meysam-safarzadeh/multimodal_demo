from rest_framework import viewsets
from .models import Dataset, DetectionResult, TrainingJob, Metric, Artifact
from .serializers import (
    DatasetSerializer, DetectionResultSerializer, TrainingJobSerializer,
    MetricSerializer, ArtifactSerializer
)

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer

class DetectionResultViewSet(viewsets.ModelViewSet):
    queryset = DetectionResult.objects.all()
    serializer_class = DetectionResultSerializer

class TrainingJobViewSet(viewsets.ModelViewSet):
    queryset = TrainingJob.objects.all()
    serializer_class = TrainingJobSerializer

class MetricViewSet(viewsets.ModelViewSet):
    queryset = Metric.objects.all()
    serializer_class = MetricSerializer

class ArtifactViewSet(viewsets.ModelViewSet):
    queryset = Artifact.objects.all()
    serializer_class = ArtifactSerializer