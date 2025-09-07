import os
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from pydantic import Field, Json, field_validator
from pydantic_settings import BaseSettings


# Load .env if it exists
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    # AWS
    aws_access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = Field(None, env="AWS_SESSION_TOKEN")
    aws_region: str = Field("eu-central-1", env="AWS_REGION")

    # S3
    s3_bucket: str = Field(..., env="S3_BUCKET")

    # Training
    epochs: int = Field(5, env="EPOCHS")
    learning_rate: float = Field(0.001, env="LEARNING_RATE")
    batch_size: int = Field(16, env="BATCH_SIZE")
    val_split: float = Field(0.2, env="VAL_SPLIT")
    workdir: str = Field("/tmp/trainer", env="WORKDIR")
    model_name: str = Field(..., env="MODEL_NAME")  # e.g. "multimodal_classification"
    output_prefix: str = Field("artifacts/", env="OUTPUT_PREFIX")
    job_id: int = Field(..., env="JOB_ID")
    assets: list[dict[str, str]] = Field(default_factory=list, env="ASSETS")
    feature_columns: list[str] = Field(default_factory=list, env="FEATURE_COLUMNS")  # comma-separated
    modality_columns: list[str] = Field(default_factory=list, env="MODALITY_COLUMNS")  # comma-separated
    target_column: str = Field(..., env="TARGET_COLUMN")
    column_types: dict[str, str] = Field(default_factory=dict, env="COLUMN_TYPES")  # JSON string
    train_pct: int = Field(80, env="TRAIN_PCT")
    test_pct: int = Field(20, env="TEST_PCT")
    early_stopping: bool = Field(True, env="EARLY_STOPPING")
    early_stopping_patience: int = Field(5, env="EARLY_STOPPING_PATIENCE")
    random_seed: int = Field(42, env="RANDOM_SEED")
    eval_steps: int = Field(1, env="EVAL_STEPS")

    # Server callback
    api_base: str = Field("http://localhost:8000", env="API_BASE")
    callback_secret: str = Field("super-long-random-string", env="CALLBACK_SECRET")
    callback_ttl: int = Field(900, env="CALLBACK_TTL")

    @field_validator("aws_access_key_id", "aws_secret_access_key")
    def _must_not_be_placeholder(cls, v):
        if v.strip().lower().startswith("your_"):
            raise ValueError(f"Invalid placeholder value: {v}")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton accessor
settings = Settings()
