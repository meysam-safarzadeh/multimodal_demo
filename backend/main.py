from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile, os, time, math

import pandas as pd

from metadata_detector import MetadataDetector

app = FastAPI(title="Multimodal Trainer API", version="0.1")

# Allow frontend (Streamlit) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Models
# -------------------------------
class TrainRequest(BaseModel):
    feature_columns: List[str]
    modality_columns: List[str]
    target_col: Optional[str]
    job_name: Optional[str] = "local-demo"

class TrainResponse(BaseModel):
    job_id: str
    status: str
    metrics: Dict[str, Any]
    model_artifact_path: str

# -------------------------------
# Helpers
# -------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively convert NaN/Inf to None so Starlette JSON can serialize."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # Pandas/NumPy aware nulls
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(csv: UploadFile = File(...)):
    """Analyze uploaded CSV with MetadataDetector."""
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await csv.read())
        tmp_path = tmp.name

    try:
        detector = MetadataDetector(tmp_path)
        summary = detector.detect()

        # Preview first 10 rows; make JSON-safe (replace NaN/Inf with None)
        preview_df = detector.df.head(10)
        preview_jsonable = _json_safe(preview_df.where(~preview_df.isna(), None).to_dict(orient="records"))

        # Also sanitize summary (defensive)
        summary_jsonable = _json_safe(summary)

        return {"summary": summary_jsonable, "preview": preview_jsonable}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    """Simulate training (replace with real ECS trigger or PyTorch training)."""
    time.sleep(1.0)  # simulate training time
    metrics = {"accuracy": 0.92, "loss": 0.13, "classes": 3}
    artifact = f"s3://demo-bucket/models/{req.job_name}.pt"  # placeholder path
    return {
        "job_id": "demo-123",
        "status": "completed",
        "metrics": metrics,
        "model_artifact_path": artifact,
    }
