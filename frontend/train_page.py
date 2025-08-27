import streamlit as st
import pandas as pd
import requests
import time
from utils import render_metrics


def render_train_page(api_url: str, csv_blob: bytes, on_back, on_home):
    st.title("⚙️ Step 2: Train Model")

    if not csv_blob:
        st.warning("⚠️ Please upload a CSV file on the previous page.")
        st.button("⬅️ Back", on_click=on_back)
        return

    # Call /detect once and cache result in session_state
    if "detect_result" not in st.session_state:
        with st.spinner("Analyzing CSV..."):
            files = {"csv": ("data.csv", csv_blob, "text/csv")}
            try:
                r = requests.post(f"{api_url}/detect", files=files, timeout=120)
                r.raise_for_status()
                st.session_state.detect_result = r.json()
            except Exception as e:
                st.error(f"Failed to analyze CSV: {e}")
                st.button("⬅️ Back", on_click=on_back)
                return

    res = st.session_state.detect_result
    summary = res.get("summary", {})
    preview = res.get("preview", [])

    # --- CSV Preview ---
    st.subheader("📄 CSV Preview")
    try:
        st.dataframe(pd.DataFrame(preview))
    except Exception:
        st.write(preview)

    # --- Metadata Summary ---
    st.subheader("🔍 Detected Metadata")
    n_rows = summary.get("n_rows", 0)
    n_columns = summary.get("n_columns", 0)
    modalities = summary.get("modalities", [])
    st.write(f"**Rows:** {n_rows}  •  **Columns:** {n_columns}")
    st.write(f"**Modalities:** {', '.join(modalities) if modalities else 'None'}")

    with st.expander("Column Types"):
        st.json(summary.get("column_types", {}))
    with st.expander("Missing Data"):
        st.json(summary.get("missing_data", {}))

    # --- Configure Training ---
    st.subheader("⚙️ Configure Training")

    suggested_features = summary.get("feature_columns", [])
    suggested_modalities = summary.get("modality_columns", [])
    suggested_targets = summary.get("target_columns_categorical", [])

    # 1) Multi-select feature columns
    feature_cols = st.multiselect(
        "Select feature columns",
        options=suggested_features,
        default=st.session_state.get("feature_cols", []),
    )
    st.session_state["feature_cols"] = feature_cols

    # 2) Multi-select modality columns
    modality_cols = st.multiselect(
        "Select modality columns",
        options=suggested_modalities,
        default=st.session_state.get("modality_cols", []),
    )
    st.session_state["modality_cols"] = modality_cols

    # 3) Target column (exclude anything chosen as a feature)
    available_targets = [c for c in suggested_targets if c not in feature_cols]
    if not available_targets:
        st.warning("No target columns available after excluding selected feature columns.")
        target_col = None
    else:
        prev_target = st.session_state.get("target_col")
        idx = available_targets.index(prev_target) if prev_target in available_targets else 0
        target_col = st.selectbox("Select target column", options=available_targets, index=idx)
        st.session_state["target_col"] = target_col

    # --- Train Button: call /train ---
    disabled = target_col is None
    if st.button("Start Training", disabled=disabled):
        with st.spinner("Training (simulated)…"):
            payload = {
                "feature_columns": feature_cols,
                "modality_columns": modality_cols,
                "target_col": target_col,
                "job_name": "local_demo_job",
            }
            try:
                r = requests.post(f"{api_url}/train", json=payload, timeout=300)
                r.raise_for_status()
                out = r.json()
                st.success("✅ Training complete!")
                st.write("**Metrics:**")
                metrics = out.get("metrics", {})
                render_metrics(metrics)
                st.write(f"**Model artifact:** {out.get('model_artifact_path')}")
                st.download_button("Download Model", data=b"dummy-model", file_name="model.pt")
            except Exception as e:
                st.error(f"Training failed: {e}")

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅️ Back", on_click=on_back)
    with col2:
        st.button("🏠 Home", on_click=on_home)
