import io
import streamlit as st
import pandas as pd
import requests
import time
from utils import render_metrics


def render_config_page(api_url: str, csv_blob: bytes, on_back, on_home):
    st.title("⚙️ Step 2: Training Configuration")

    # If they uploaded a new dataset, we need to POST /datasets/upload_and_detect
    if not st.session_state.get("dataset"):
        with st.spinner("Uploading files and running metadata detection..."):
            try:
                # Build multipart payload
                files = []
                # CSV
                files.append(("csv_file", ("metadata.csv", csv_blob, "text/csv")))
                # Optional: additional data files saved from Page 1
                # Expecting st.session_state["data_files"] to be a list of Streamlit UploadedFile
                for f in st.session_state.get("data_files", []):
                    fname = getattr(f, "name", "file.bin")
                    mime = getattr(f, "type", "application/octet-stream")
                    files.append(("data_files", (fname, f.getvalue(), mime)))

                data = {
                    "name": st.session_state.get("dataset_name", "demo-dataset"),
                }

                r = requests.post(
                    f"{api_url}/datasets/upload_and_detect/",
                    data=data,
                    files=files,
                    timeout=300,
                )
                r.raise_for_status()
                ds = r.json()

                # Cache dataset + id
                st.session_state.dataset = ds
                st.session_state.dataset_id = ds.get("id")

                # Unify detection summary whether you kept split models (ds["detection"])
                # or merged fields directly on the dataset (ds[...] without "detection").
                det = ds.get("detection") or ds
                st.session_state.detect_result = {
                    "summary": {
                        "n_rows": det.get("n_rows", 0),
                        "n_columns": det.get("n_columns", 0),
                        "modalities": det.get("modalities", []),
                        "column_types": det.get("column_types", {}),
                        "missing_data": det.get("missing_data", {}),
                        "feature_columns": det.get("feature_columns", []),
                        "modality_columns": det.get("modality_columns", []),
                        "target_columns_categorical": det.get("target_columns_categorical", []),
                    }
                }

            except Exception as e:
                st.error(f"Upload & detection failed: {e}")
                st.button("⬅️ Back", on_click=on_back)
                return
    
    ds = st.session_state.get("dataset", {})
    res = st.session_state.detect_result
    summary = res.get("summary", {})

    # --- CSV Preview ---
    st.subheader("📄 CSV Preview")
    # Case 1: local CSV just uploaded on page 1
    if st.session_state.get("csv_blob"):
        try:
            df_preview = pd.read_csv(io.BytesIO(st.session_state["csv_blob"]), nrows=50)
            st.dataframe(df_preview, width="stretch")
        except Exception as e:
            st.warning(f"Could not render local CSV preview: {e}")

    # Case 2: existing dataset selected (no local CSV)
    elif st.session_state.get("dataset_id"):
        rid = st.session_state["dataset_id"]
        with st.spinner("Fetching preview..."):
            try:
                resp = requests.get(f"{api_url}/datasets/{rid}/preview/", timeout=60)
                if resp.status_code == 202:
                    st.info("CSV is still uploading. Preview will be available shortly.")
                else:
                    resp.raise_for_status()
                    prev = resp.json()  # {"columns": [...], "rows": [...], "n": 50}
                    df_preview = pd.DataFrame(prev.get("rows", []))
                    if df_preview.empty:
                        st.info("No preview available yet.")
                    else:
                        st.dataframe(df_preview, width="stretch")
            except Exception as e:
                st.warning(f"Could not load preview from API: {e}")

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

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅️ Back", on_click=on_back, use_container_width=True)
    with col2:
        st.button("🏠 Home", on_click=on_home, use_container_width=True)
