from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union
import pandas as pd


@dataclass
class ClassificationMetrics:
    accuracy: float = 0.0
    loss: float = 0.0
    val_accuracy: float = 0.0
    val_loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)


@dataclass
class TrainingConfiguration:
    """Dataclass for storing model data."""
    learning_rate: float
    epochs: int
    batch_size: int
    early_stopping: bool
    early_stopping_patience: int
    random_seed: int
    eval_steps: int


@dataclass
class DLTrainingParameters:
    """Dataclass for storing training parameters."""
    model_name: str
    assets_paths: List[Dict[str, str]]
    training_job_id: int
    validation_split: float
    configuration: TrainingConfiguration
    target_column: str
    column_types: Dict[str, str]
    feature_columns: Optional[List[str]] = []
    modality_columns: Optional[List[str]] = []


@dataclass
class TrainingReport:
    metrics: ClassificationMetrics
    summary: str = ""
    duration: float = 0.0
    logs: List[Dict] = field(default_factory=list)
    artifacts: List[Dict] = field(default_factory=list)

    @property
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Artifacts:
    trained_model: Any
    model_type: Literal['pkl', 'pt', 'bin']
    id2label: Optional[Dict[str, str]] = None