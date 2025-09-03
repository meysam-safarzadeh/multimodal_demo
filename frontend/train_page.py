import time
import requests
import pandas as pd
import streamlit as st
from utils import render_training_curves

def start_training(api_url: str, job_id: int):
    r = requests.post(f"{api_url}/training_jobs/{job_id}/start/", timeout=30)
    r.raise_for_status()
    return r.json()

def get_job(api_url: str, job_id: int):
    r = requests.get(f"{api_url}/training_jobs/{job_id}/", timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_metrics(api_url: str, job_id: int):
    r = requests.get(f"{api_url}/training_jobs/{job_id}/metrics/", timeout=30)
    r.raise_for_status()
    return r.json()

def poll_until_done(api_url: str, job_id: int, max_seconds: int = 900, interval: float = 3.0):
    start = time.time()
    while True:
        job = get_job(api_url, job_id)
        status = job.get("status", "UNKNOWN")
        yield status  # stream status to the UI
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        if time.time() - start > max_seconds:
            raise TimeoutError("Polling timed out.")
        time.sleep(interval)

def render_train_results_page(api_url: str, on_back, on_home):
    st.title("🧪 Step 3: Train & Results")

    job_id = st.session_state.get("training_job_id")
    if not job_id:
        st.warning("No training job found. Please create a job on Step 2.")
        st.button("⬅️ Back", on_click=on_back)
        return

    # Show basic job info
    try:
        job = get_job(api_url, job_id)
    except Exception as e:
        st.error(f"Could not load job: {e}")

    # Start Training button
    colA, colB, colC = st.columns([1,1,2])
    with colA:
        if st.button("▶️ Start Training", width="stretch"):
            try:
                start_training(api_url, job_id)
                st.success("Training started.")
                st.session_state[f"job_started_{job_id}"] = True
            except Exception as e:
                st.error(f"Failed to start training: {e}")

    with colB:
        if st.button("🔄 Refresh status", width="stretch"):
            st.rerun()

    # Live polling (only after start)
    if st.session_state.get(f"job_started_{job_id}"):
        status_placeholder = st.empty()
        try:
            for status in poll_until_done(api_url, job_id, max_seconds=1800, interval=3.0):
                status_placeholder.info(f"Status: **{status}**")
            status_placeholder.success(f"Status: **{status}**")
        except TimeoutError:
            status_placeholder.warning("Still running… (you can refresh later)")
        except Exception as e:
            status_placeholder.error(f"Training failed: {e}")

    # Metrics
    st.subheader("📈 Metrics")
    try:
        m = fetch_metrics(api_url, job_id)
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m.get('accuracy', '—')}")
        cols[1].metric("Loss", f"{m.get('loss', '—')}")
        cols[2].metric("Val Acc", f"{m.get('val_accuracy', '—')}")
        cols[3].metric("Val Loss", f"{m.get('val_loss', '—')}")

        # fancy plots
        st.subheader("📉 Training Curves")
        render_training_curves(m)
    
    except Exception:
        st.caption("No metrics yet.")

    # Nav
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅️ Back", on_click=on_back, width="stretch")
    with c2:
        st.button("🏠 Home", on_click=on_home, width="stretch")
