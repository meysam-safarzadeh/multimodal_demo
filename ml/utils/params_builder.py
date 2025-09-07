import logging
from typing import Dict, List

from config import settings
from models.schemas import DLTrainingParameters, TrainingConfiguration

logger = logging.getLogger("trainer")

def build_training_params(local_assets: List[Dict]) -> DLTrainingParameters:
    """
    Build DLTrainingParameters using values solely from ml.config.settings and resolved local assets.
    """
    logger.info("[params_builder] Building DLTrainingParameters from settings")

    cfg = TrainingConfiguration(
        learning_rate=float(settings.learning_rate),
        epochs=int(settings.epochs),
        batch_size=int(settings.batch_size),
        early_stopping=bool(settings.early_stopping),
        early_stopping_patience=int(settings.early_stopping_patience),
        random_seed=int(settings.random_seed),
        eval_steps=int(settings.eval_steps),
    )

    params = DLTrainingParameters(
        model_name=settings.model_name,
        feature_columns=list(settings.feature_columns or []),
        modality_columns=list(settings.modality_columns or []),
        column_types=dict(settings.column_types or {}),
        target_column=str(settings. target_column),
        validation_split=float(settings.val_split),
        assets_paths=local_assets,
        training_job_id=int(settings.job_id),
        configuration=cfg,
    )

    logger.info("[params_builder] Params ready")
    return params
