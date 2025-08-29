from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import Dataset, DetectionResult, TrainingJob, Metric, Artifact
from .serializers import (
    DatasetSerializer, DetectionResultSerializer, TrainingJobSerializer,
    MetricSerializer, ArtifactSerializer
)
from services.training_services import start_training_background


class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer

class DetectionResultViewSet(viewsets.ModelViewSet):
    queryset = DetectionResult.objects.all()
    serializer_class = DetectionResultSerializer

class TrainingJobViewSet(viewsets.ModelViewSet):
    queryset = TrainingJob.objects.all()
    serializer_class = TrainingJobSerializer

    @action(detail=True, methods=["post"], url_path="start")
    def start(self, request, pk=None):
        # Optional: prevent duplicate starts
        job = self.get_object()
        if job.status in [TrainingJob.Status.RUNNING]:
            return Response({"detail": "Job already running."}, status=status.HTTP_409_CONFLICT)

        start_training_background(job.id)
        return Response({"message": "Training started", "job_id": job.id}, status=status.HTTP_202_ACCEPTED)


class MetricViewSet(viewsets.ModelViewSet):
    queryset = Metric.objects.all()
    serializer_class = MetricSerializer

class ArtifactViewSet(viewsets.ModelViewSet):
    queryset = Artifact.objects.all()
    serializer_class = ArtifactSerializer