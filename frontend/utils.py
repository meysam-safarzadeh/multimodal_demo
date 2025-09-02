import requests
import streamlit as st
import pandas as pd


def render_metrics(metrics: dict, title: str = "📊 Training Metrics") -> None:
    """
    Nicely render training metrics in Streamlit.

    Args:
        metrics (dict): Dictionary of metric name -> value
        title (str): Optional title to display above metrics
    """
    if not metrics:
        st.info("No metrics available.")
        return

    st.subheader(title)

    # Display key metrics as Streamlit metrics (cards)
    cols = st.columns(len(metrics))
    for (key, val), col in zip(metrics.items(), cols):
        # Format floats nicely
        if isinstance(val, (int, float)):
            display_val = f"{val:.4f}" if isinstance(val, float) else str(val)
        else:
            display_val = str(val)
        col.metric(label=key.capitalize(), value=display_val)


def create_training_job(api_url: str) -> int:
    """Create a TrainingJob from values in session_state. Returns job_id."""
    ds_id = st.session_state.get("dataset_id")
    if not ds_id:
        raise ValueError("No dataset selected or created.")

    payload = {
        "dataset": ds_id,
        "job_name": st.session_state.get("job_name", "local_demo_job"),
        "feature_columns": st.session_state.get("feature_cols", []),
        "modality_columns": st.session_state.get("modality_cols", []),
        "target_col": st.session_state.get("target_col"),
        "epochs": st.session_state.get("epochs", 10),
        "learning_rate": st.session_state.get("learning_rate", 1e-3),
        "train_pct": st.session_state.get("train_pct", 80),
        "test_pct": st.session_state.get("test_pct", 20),
        "status": "PENDING",
    }
    r = requests.post(f"{api_url}/training_jobs/", json=payload, timeout=60)
    r.raise_for_status()
    job = r.json()
    return job["id"]

