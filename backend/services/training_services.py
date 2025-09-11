# backend/services/training_services.py
import threading
import time
from typing import Optional
from django.utils import timezone
from django.db import transaction
from django.conf import settings
from registry.models import TrainingJob, Artifact, TrainingMetrics


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

        if _ecs_configured():
            task_arn = ecs_run_task(job_id)
            _ecs_wait_until_stopped(task_arn)
        else:
            _local_dummy_training(job_id)

        Artifact.objects.create(job_id=job_id, kind="model",
                                s3_uri=f"s3://{settings.S3_BUCKET}/models/job-{job_id}.pt")

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

def ecs_run_task(job_id: int) -> str:
    import boto3
    ecs = boto3.client(
        "ecs",
        region_name=settings.ECS["AWS_REGION"],
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )
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
                    "environment": [
                        {"name": "MODEL", "value": "multimodal"},
                        {"name": "JOB_ID", "value": str(job_id)},
                        {"name": "S3_BUCKET", "value": settings.S3_BUCKET},
                        {"name": "API_BASE", "value": settings.API_BASE},
                        {"name": "CALLBACK_SECRET", "value": settings.CALLBACK_SECRET},
                        {"name": "CALLBACK_TTL", "value": str(settings.CALLBACK_TTL)},
                        {"name": "AWS_ACCESS_KEY_ID", "value": settings.AWS_ACCESS_KEY_ID},
                        {"name": "AWS_SECRET_ACCESS_KEY", "value": settings.AWS_SECRET_ACCESS_KEY},
                        {"name": "epochs", "value": str(settings.EPOCHS)},
                        {"name": "learning_rate", "value": str(settings.LEARNING_RATE)},
                        {"name": "batch_size", "value": str(settings.BATCH_SIZE)},
                        {"name": "val_split", "value": str(settings.VAL_SPLIT)},
                        {"name": "feature_columns", "value": ",".join(settings.FEATURE_COLUMNS or [])},
                        {"name": "modality_columns", "value": ",".join(settings.MODALITY_COLUMNS or [])},
                        {"name": "target_column", "value": settings.TARGET_COLUMN},
                        {"name": "column_types", "value": settings.COLUMN_TYPES or "{}"},
                        {"name": "early_stopping", "value": str(settings.EARLY_STOPPING)},
                        {"name": "early_stopping_patience", "value": str(settings.EARLY_STOPPING_PATIENCE)},
                        {"name": "random_seed", "value": str(settings.RANDOM_SEED)},
                        {"name": "eval_steps", "value": str(settings.EVAL_STEPS)},
                        {"name": "WORKDIR", "value": "/tmp/trainer"},
                        {"name": "ASSETS", "value": str(settings.ASSETS)},
                        {"name": "OUTPUT_PREFIX", "value": "artifacts/"},
                        {"name": "TRAIN_PCT", "value": str(settings.TRAIN_PCT)},
                        {"name": "TEST_PCT", "value": str(settings.TEST_PCT)}
                    ],
                }
            ]
        },
    )
    failures = resp.get("failures") or []
    if failures:
        raise RuntimeError(f"ECS run_task failed: {failures}")
    task_arn = resp["tasks"][0]["taskArn"]
    TrainingJob.objects.filter(id=job_id).update(ecs_task_arn=task_arn)
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
            break
        if waited >= timeout:
            raise TimeoutError("ECS task did not stop within timeout")
