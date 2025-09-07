import os
import logging
from pathlib import Path
from typing import Any, List, Dict, Tuple
from urllib.parse import urlparse
import boto3
import botocore
from config import settings
import io
import json


logger = logging.getLogger("trainer")

def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        aws_session_token=settings.aws_session_token,
        region_name=settings.aws_region,
    )


def _parse_s3_ref(asset: Dict, default_bucket: str | None) -> Tuple[str, str]:
    """
    Accepts any of:
      - {"s3_uri": "s3://bucket/key"}
      - {"bucket": "bucket", "key": "key"}
      - {"s3_uri": "datasets/17/file.csv"}  -> uses default_bucket (if provided)
    Returns (bucket, key). Raises ValueError if insufficient info.
    """
    if "s3_uri" in asset:
        uri = asset["s3_uri"]
        if uri.startswith("s3://"):
            u = urlparse(uri)
            return u.netloc, u.path.lstrip("/")
        # treat as key relative to default bucket
        if default_bucket:
            return default_bucket, uri.lstrip("/")
        raise ValueError(f"s3_uri lacks scheme and no default bucket provided: {uri}")

    if "bucket" in asset and "key" in asset:
        return asset["bucket"], asset["key"].lstrip("/")

    raise ValueError(f"Asset must contain s3_uri OR (bucket & key): {asset}")


def _is_prefix(key: str) -> bool:
    # Simple heuristic: if endswith '/', treat as prefix
    return key.endswith("/")


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def download_assets_from_s3(assets: List[Dict], workdir: Path, default_bucket: str | None = None) -> List[Dict]:
    """
    Download a list of assets to local workdir.
    Supports both single-object and prefix ("folder") downloads.

    Asset format examples:
      {"key": "train_file",  "s3_uri": "s3://my-bucket/datasets/17/metadata.csv"}
      {"key": "train_file",  "s3_uri": "datasets/17/metadata.csv"}  # needs default_bucket
      {"key": "train_folder","s3_uri": "s3://my-bucket/datasets/17/data/"}  # prefix => sync folder
      {"key": "train_folder","bucket": "my-bucket","key": "datasets/17/data/"}  # prefix

    Returns a list of:
      - for single file: {"key": <str>, "local_path": "/path/to/file.csv"}
      - for prefix:      {"key": <str>, "local_path":  "/path/to/data"}
    """
    s3 = _s3_client()
    results: List[Dict] = []

    for asset in assets:
        logical_key = asset.get("key", "asset")
        bucket, key = _parse_s3_ref(asset, default_bucket)
        logger.info(f"[S3] Resolving asset '{logical_key}': s3://{bucket}/{key}")

        if _is_prefix(key):
            # Download all objects under this prefix into a local dir
            local_dir = workdir / key.rstrip("/").split("/")[-1]
            local_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[S3] Sync prefix s3://{bucket}/{key} -> {local_dir}")

            paginator = s3.get_paginator("list_objects_v2")
            any_found = False
            for page in paginator.paginate(Bucket=bucket, Prefix=key):
                for obj in page.get("Contents", []):
                    any_found = True
                    obj_key = obj["Key"]
                    # keep folder structure under local_dir
                    rel = obj_key[len(key):]  # strip the prefix
                    if not rel or rel.endswith("/"):
                        continue
                    dst = local_dir / rel
                    _ensure_dir(dst)
                    logger.info(f"[S3]   downloading {obj_key} -> {dst}")
                    s3.download_file(bucket, obj_key, str(dst))
            if not any_found:
                raise FileNotFoundError(f"No objects found under prefix s3://{bucket}/{key}")
            results.append({"key": logical_key, "local_path": str(local_dir)})

        else:
            # Single object download
            filename = os.path.basename(key)
            local_path = workdir / filename
            _ensure_dir(local_path)
            logger.info(f"[S3] Download file s3://{bucket}/{key} -> {local_path}")
            try:
                s3.download_file(bucket, key, str(local_path))
            except botocore.exceptions.ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                    raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}") from e
                raise
            results.append({"key": logical_key, "local_path": str(local_path)})

    return results


def upload_artifacts_to_s3(
    bucket: str,
    artifacts: Dict[str, object],
    job_id: int,
    prefix: str = "artifacts"
) -> List[Dict[str, str]]:
    """
    Uploads artifacts to s3://{bucket}/{prefix}/{job_id}/
    - id2label: dict → JSON
    - trained_model: torch state_dict() → .pt
    Returns list of {"key": ..., "s3_uri": ...}
    """
    import torch
    
    s3 = _s3_client()
    out: List[Dict[str, str]] = []
    base_prefix = f"{prefix.rstrip('/')}/{job_id}"

    # --- Upload id2label.json ---
    id2label_key = f"{base_prefix}/id2label.json"
    id2label_bytes = json.dumps(artifacts.id2label).encode("utf-8")
    s3.upload_fileobj(
        Fileobj=io.BytesIO(id2label_bytes),
        Bucket=bucket,
        Key=id2label_key,
        ExtraArgs={"ContentType": "application/json"},
    )
    out.append({"key": "id2label", "s3_uri": f"s3://{bucket}/{id2label_key}"})

    # --- Upload model.pt ---
    model_key = f"{base_prefix}/model.pt"
    buffer = io.BytesIO()
    torch.save(artifacts.trained_model, buffer)
    buffer.seek(0)
    s3.upload_fileobj(
        Fileobj=buffer,
        Bucket=bucket,
        Key=model_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )
    out.append({"key": "trained_model", "s3_uri": f"s3://{bucket}/{model_key}"})

    return out


def _parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Not a valid s3 uri: {uri}")
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1]
