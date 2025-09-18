# backend/services/training_services.py
import json
import threading
import time
from typing import Optional
from django.utils import timezone
from django.db import transaction
from django.conf import settings
from registry.models import Dataset, TrainingArtifacts, TrainingJob


def start_training_background(job_id: int) -> None:
    """Spawn a background thread so the API can return immediately."""
    t = threading.Thread(target=_run_training, args=(job_id,), daemon=True)
    t.start()


def _run_training(job_id: int) -> None:
    try:
        with transaction.atomic():
            job = TrainingJob.objects.select_for_update().get(id=job_id)
            job.status = TrainingJob.Status.RUNNING
            job.started_at = timezone.now()
            job.save()

        task_arn = ecs_run_task(job)
        exitcode = _ecs_wait_until_stopped(task_arn)

        if exitcode != 0:
            raise RuntimeError(f"Training job in ECS failed with exit code {exitcode}")

        TrainingJob.objects.filter(id=job_id).update(
            status=TrainingJob.Status.COMPLETED, ended_at=timezone.now()
        )
    except Exception:
        TrainingJob.objects.filter(id=job_id).update(
            status=TrainingJob.Status.FAILED, ended_at=timezone.now()
        )
        raise

def _local_dummy_training(job_id: int) -> None:
    time.sleep(3)

def _ecs_configured() -> bool:
    ecs = settings.ECS
    return all([ecs["AWS_REGION"], ecs["CLUSTER"], ecs["TASK_DEFINITION"],
                ecs["SUBNETS"], ecs["SECURITY_GROUPS"]])

def ecs_run_task(job: TrainingJob) -> str:
    import boto3
    ecs = boto3.client(
        "ecs",
        region_name=settings.ECS["AWS_REGION"],
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )
    # Retrieve the dataset's CSV S3 URI using job.dataset_id
    dataset = Dataset.objects.get(id=job.dataset.pk)
    assets = [
        {"key": "train_file", "s3_uri": dataset.csv_s3_uri},
        {"key": "train_folder", "s3_uri": dataset.data_s3_uri},
    ]

    env = [
        {"name": "MODEL", "value": "multimodal"},
        {"name": "JOB_ID", "value": str(job.pk)},
        {"name": "S3_BUCKET", "value": settings.S3_BUCKET},
        {"name": "API_BASE", "value": settings.API_BASE},
        {"name": "CALLBACK_SECRET", "value": settings.CALLBACK_SECRET},
        {"name": "CALLBACK_TTL", "value": str(settings.CALLBACK_TTL)},
        {"name": "AWS_ACCESS_KEY_ID", "value": settings.AWS_ACCESS_KEY_ID},
        {"name": "AWS_SECRET_ACCESS_KEY", "value": settings.AWS_SECRET_ACCESS_KEY},

        # Hyperparams
        {"name": "EPOCHS", "value": str(job.epochs)},
        {"name": "LEARNING_RATE", "value": str(job.learning_rate)},
        {"name": "BATCH_SIZE", "value": str(job.batch_size)},
        {"name": "VALIDATION_SPLIT", "value": str(job.val_split)},

        # Schema: send JSON
        {"name": "FEATURE_COLUMNS", "value": json.dumps(job.feature_columns or [])},
        {"name": "MODALITY_COLUMNS", "value": json.dumps(job.modality_columns or [])},
        {"name": "TARGET_COLUMN", "value": job.target_col or ""},
        {"name": "COLUMN_TYPES", "value": json.dumps(dataset.column_types or {})},

        {"name": "MODEL_NAME", "value": "multimodal_classification"},
        {"name": "WORKDIR", "value": "/tmp/trainer"},

        # IMPORTANT: match Settings field
        {"name": "ASSETS", "value": json.dumps(assets)},
    ]

    resp = ecs.run_task(
        cluster=settings.ECS["CLUSTER"],
        taskDefinition=settings.ECS["TASK_DEFINITION"],
        launchType="FARGATE",
        platformVersion=settings.ECS["PLATFORM_VERSION"],
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": settings.ECS["SUBNETS"],
                "securityGroups": settings.ECS["SECURITY_GROUPS"],
                "assignPublicIp": settings.ECS["ASSIGN_PUBLIC_IP"],
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": settings.ECS["CONTAINER_NAME"],
                    "environment": env,
                }
            ]
        },
    )

    failures = resp.get("failures") or []
    if failures:
        raise RuntimeError(f"ECS run_task failed: {failures}")
    task_arn = resp["tasks"][0]["taskArn"]
    TrainingJob.objects.filter(id=job.pk).update(ecs_task_arn=task_arn)
    return task_arn

def _ecs_wait_until_stopped(task_arn: str, poll=5, timeout=3600):
    import time, boto3
    ecs = boto3.client("ecs", region_name=settings.ECS["AWS_REGION"])
    cluster = settings.ECS["CLUSTER"]
    waited = 0
    while True:
        time.sleep(poll)
        waited += poll
        desc = ecs.describe_tasks(cluster=cluster, tasks=[task_arn])
        tasks = desc.get("tasks", [])
        if tasks and tasks[0].get("lastStatus") == "STOPPED":
            exitcode = tasks[0].get("containers", [{}])[0].get("exitCode")
            break
        if waited >= timeout:
            raise TimeoutError("ECS task did not stop within timeout")
    
    return exitcode
