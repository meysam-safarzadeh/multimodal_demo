# backend/services/training_services.py
import threading
import time
from typing import Optional
from django.utils import timezone
from django.db import transaction

from registry.models import TrainingJob, Metric, Artifact

def _run_training(job_id: int) -> None:
    """Do the training work (dummy implementation)."""
    try:
        with transaction.atomic():
            job = TrainingJob.objects.select_for_update().get(id=job_id)
            job.status = TrainingJob.Status.RUNNING
            job.started_at = timezone.now()
            job.save()

        # TODO: replace this block with actual training call
        # e.g., subprocess.run(["python", "ML/multimodal_model/train.py", "--job-id", str(job_id)], check=True)
        time.sleep(3)  # simulate training

        # Write some example metrics
        Metric.objects.create(job_id=job_id, key="accuracy", value_num=0.9231)
        Metric.objects.create(job_id=job_id, key="loss", value_num=0.1289)
        Metric.objects.create(job_id=job_id, key="classes", value_num=3)

        # Save a model artifact location (could be local path or S3 URI)
        Artifact.objects.create(job_id=job_id, kind="model", s3_uri=f"s3://demo-bucket/models/job-{job_id}.pt")

        # Mark complete
        TrainingJob.objects.filter(id=job_id).update(
            status=TrainingJob.Status.COMPLETED,
            ended_at=timezone.now()
        )
    except Exception:
        TrainingJob.objects.filter(id=job_id).update(
            status=TrainingJob.Status.FAILED,
            ended_at=timezone.now()
        )
        raise

def start_training_background(job_id: int) -> None:
    """Spawn a background thread so the API can return immediately."""
    t = threading.Thread(target=_run_training, args=(job_id,), daemon=True)
    t.start()
