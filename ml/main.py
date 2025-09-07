import os
import sys
import json
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import boto3
import psycopg2
from psycopg2.extras import Json
import requests
from utils.authentication import make_token
from utils.params_builder import build_training_params
from utils.storage import download_assets_from_s3, upload_artifacts_to_s3
from config import settings


# load .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainer")


# DB persistence
def write_training_report(job_id: int, 
                          training_report: Dict[str, Any], 
                          artifact_uris: List[Dict[str, str]]) -> None:
    logger.info(f"Writing training report for job_id={job_id}")

    job_id = settings.job_id
    payload = {
        "accuracy": training_report.metrics.accuracy,
        "loss": training_report.metrics.loss,
        "val_accuracy": training_report.metrics.val_accuracy,
        "val_loss": training_report.metrics.val_loss,
        "logs": training_report.logs,
    }

    token = make_token(job_id, settings.callback_secret, settings.callback_ttl)
    url = f"{settings.api_base}/training_jobs/{job_id}/metrics/"
    resp = requests.patch(url, json=payload, headers={"X-Callback-Token": token}, timeout=10)
    resp.raise_for_status()
    
    # Update logger
    if resp.status_code == 200:
        logger.info(f"Successfully updated training metrics for job_id={job_id}")
    else:
        logger.error(f"Failed to update training metrics for job_id={job_id}: {resp.status_code} {resp.text}")


# ---------------------------------------------------------------------
# Dynamic import of train() from selected model
# ---------------------------------------------------------------------
def import_model_train(model_name: str):
    """
    Dynamically import models/<model_name>/train.py and return its train() function.
    """
    module_path = f"models.{model_name}.train"
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        logger.error(f"Could not import model {model_name}: {e}")
        sys.exit(1)

    if not hasattr(module, "train"):
        logger.error(f"Model {model_name} has no train() function in train.py")
        sys.exit(1)

    return module.train


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main():
    job_id = settings.job_id
    model_name = settings.model_name
    bucket = settings.s3_bucket

    logger.info(f"=== Starting training job {job_id} with model {model_name} ===")

    workdir = Path(settings.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Download assets
    local_assets = download_assets_from_s3(settings.assets, workdir)

    # Import and run train()
    train_fn = import_model_train(model_name)
    params = build_training_params(local_assets)

    logger.info("Running training...")
    training_report, artifacts = train_fn(params)
    logger.info("Training finished.")

    # Upload artifacts
    artifact_uris = upload_artifacts_to_s3(bucket, artifacts, job_id)

    # Save report
    write_training_report(job_id, training_report, artifact_uris)

    logger.info(f"=== Job {job_id} finished successfully ===")


if __name__ == "__main__":
    main()
