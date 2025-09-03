import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import urlparse
import boto3
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from metadata_detector import MetadataDetector
from registry import settings
from .models import Dataset, TrainingJob, Artifact, TrainingMetrics
from .serializers import (
    DatasetSerializer, TrainingJobSerializer,
    TrainingMetricsSerializer, ArtifactSerializer
)
from services.training_services import start_training_background
import pandas as pd
import io
import json
from registry.utils import _parse_s3_uri


s3 = boto3.client("s3")

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer

    @action(detail=False, methods=["post"])
    def upload_and_detect(self, request):
        csv_file = request.FILES.get("csv_file")
        data_files = request.FILES.getlist("data_files")  # can be empty or multiple
        dataset_name = request.data.get("name", "unnamed-dataset")

        if not csv_file:
            return Response({"error": "CSV file required"}, status=status.HTTP_400_BAD_REQUEST)

        # --- persist CSV to a temp file (used for detection + S3) ---
        csv_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        for chunk in csv_file.chunks():
            csv_tmp.write(chunk)
        csv_tmp.flush(); csv_tmp.close()
        csv_tmp_path = csv_tmp.name

        # --- run detection immediately on temp CSV ---
        detector = MetadataDetector(csv_tmp_path)
        detection_summary = detector.detect()

        # --- persist data files to temp paths BEFORE starting background thread ---
        saved_files = []  # list of dicts with path/name/content_type
        for f in data_files:
            suffix = Path(f.name).suffix or ""
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            for chunk in f.chunks():
                tmpf.write(chunk)
            tmpf.flush(); tmpf.close()
            saved_files.append({
                "path": tmpf.name,
                "name": f.name,
                "content_type": getattr(f, "content_type", "application/octet-stream")
            })

        # --- create DB rows (Dataset + one-to-one DetectionResult) ---
        dataset = Dataset.objects.create(
            name=dataset_name, 
            csv_s3_uri="", 
            data_s3_uri="",
            modalities=detection_summary["modalities"],
            feature_columns=detection_summary["feature_columns"],
            modality_columns=detection_summary["modality_columns"],
            target_columns_categorical=detection_summary["target_columns_categorical"],
            column_types=detection_summary["column_types"],
            missing_data=detection_summary["missing_data"],
            n_rows=detection_summary["n_rows"],
            n_columns=detection_summary["n_columns"],
        )

        # --- background uploader uses persisted temp paths ---
        def upload_to_s3(csv_tmp_path: str, saved_files: list, dataset_id: int):
            bucket = settings.S3_BUCKET  # set this in settings/.env
            try:
                # CSV
                csv_key = f"datasets/{dataset_id}/metadata.csv"
                s3.upload_file(csv_tmp_path, bucket, csv_key,
                               ExtraArgs={"ContentType": "text/csv"})
                csv_uri = f"s3://{bucket}/{csv_key}"

                # Data files
                data_uri = ""
                if saved_files:
                    prefix = f"datasets/{dataset_id}/data/"
                    for item in saved_files:
                        key = prefix + item["name"]
                        s3.upload_file(item["path"], bucket, key,
                                       ExtraArgs={"ContentType": item["content_type"]})
                    data_uri = f"s3://{bucket}/{prefix}"

                # Save URIs
                Dataset.objects.filter(id=dataset_id).update(
                    csv_s3_uri=csv_uri, data_s3_uri=data_uri
                )
            finally:
                # clean up temp files
                try:
                    os.remove(csv_tmp_path)
                except Exception:
                    pass
                for item in saved_files:
                    try:
                        os.remove(item["path"])
                    except Exception:
                        pass

        threading.Thread(
            target=upload_to_s3,
            args=(csv_tmp_path, saved_files, dataset.id),
            daemon=True
        ).start()

        # respond immediately
        return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=["get"], url_path="preview")
    def preview(self, request, pk=None):
        """
        Returns a small preview of the dataset CSV:
        {
          "columns": [...],
          "rows": [ {col: val, ...}, ... ],
          "n": 25
        }
        """
        try:
            ds = self.get_object()
            if not ds.csv_s3_uri:
                return Response({"detail": "CSV not uploaded yet."}, status=202)

            bucket, key = _parse_s3_uri(ds.csv_s3_uri)

            # Try S3 Select first (efficient for large files)
            try:
                resp = s3.select_object_content(
                    Bucket=bucket,
                    Key=key,
                    ExpressionType="SQL",
                    Expression="SELECT * FROM s3object s LIMIT 25",
                    InputSerialization={"CSV": {"FileHeaderInfo": "USE"}, "CompressionType": "NONE"},
                    OutputSerialization={"CSV": {}},
                )
                csv_buf = io.StringIO()
                for event in resp["Payload"]:
                    if "Records" in event:
                        csv_buf.write(event["Records"]["Payload"].decode("utf-8"))
                csv_buf.seek(0)
                df = pd.read_csv(csv_buf)
            except Exception:
                # Fallback: stream first bytes and let pandas read nrows
                obj = s3.get_object(Bucket=bucket, Key=key)
                # reading all body is fine for small/medium CSV; for huge, use smart chunking
                body = obj["Body"].read(5 * 1024 * 1024)  # 5MB cap just in case
                df = pd.read_csv(io.BytesIO(body), nrows=25)

            # Convert to a lightweight JSON
            rows = json.loads(df.head(25).to_json(orient="records"))
            return Response({"columns": list(df.columns), "rows": rows, "n": len(rows)}, status=200)

        except Exception as e:
            return Response({"detail": f"Failed to build preview: {e}"}, status=400)


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
    
    @action(detail=True, methods=["get", "patch"], url_path="metrics")
    def metrics(self, request, pk=None):
        job = self.get_object()
        try:
            tm = job.metrics
        except TrainingMetrics.DoesNotExist:
            # create empty row lazily if you like
            tm = TrainingMetrics.objects.create(job=job)

        if request.method.lower() == "get":
            return Response(TrainingMetricsSerializer(tm).data)

        # PATCH allows partial updates (e.g., streaming logs)
        ser = TrainingMetricsSerializer(tm, data=request.data, partial=True)
        ser.is_valid(raise_exception=True)
        ser.save()
        return Response(ser.data, status=status.HTTP_200_OK)


class ArtifactViewSet(viewsets.ModelViewSet):
    queryset = Artifact.objects.all()
    serializer_class = ArtifactSerializer