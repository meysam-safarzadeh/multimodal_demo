from typing import List, Dict, Tuple
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import re
import pandas as pd


from typing import Optional

def get_acc_and_confmatrix(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None
) -> tuple[float, List[List[int]]]:
    """
    Args:
        outputs: Tensor, shape [batch, num_classes] (raw logits)
        targets: Tensor, shape [batch] (int labels)
        num_classes: (optional) number of classes, needed for confusion matrix in edge cases
    Returns:
        accuracy: float
        confusion: ndarray (num_classes x num_classes)
    """
    # Get predicted class (max logit index)
    preds = outputs.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    accuracy = (preds == targets_np).mean()

    if num_classes is None:
        num_classes = max(preds.max(), targets_np.max()) + 1

    confusion = confusion_matrix(targets_np, preds, labels=np.arange(num_classes))
    return accuracy, confusion.tolist()


def clean_text(text: str) -> str:
    """
    Cleans a text string by:
      - Removing URLs
      - Removing emojis and non-ASCII characters
      - Collapsing extra whitespace
      - Trimming leading/trailing whitespace
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove emojis and non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove non-English characters (only allow a-z, A-Z, 0-9, space, basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9\s\.,;:!\?\-\(\)\[\]\"\'\/]", "", text)
    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Trim leading/trailing whitespace
    return text.strip()


def clean_text_columns(
    df: pd.DataFrame,
    selected_columns: List[str],
    column_types: Dict[str, str]
) -> pd.DataFrame:
    """Cleans columns of type 'text' within the selected columns of a DataFrame."""
    df_clean = df.copy()
    for col in selected_columns:
        if column_types.get(col) == "text" and col in df_clean.columns:
            # Ensure all values are strings, then clean
            df_clean[col] = df_clean[col].astype(str).apply(clean_text)
    return df_clean


def precision_recall_f1_from_cm(conf_mat: List[List[int]]) -> Tuple[List[float], List[float], List[float]]:
    # TP, FP, FN
    TP = np.diag(conf_mat)
    FP = np.sum(conf_mat, axis=0) - TP
    FN = np.sum(conf_mat, axis=1) - TP

    # Precision, Recall, F1 per class
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # Convert to list
    precision = [float(x) for x in precision]
    recall = [float(x) for x in recall]
    f1 = [float(x) for x in f1]

    return precision, recall, f1