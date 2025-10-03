import io
import streamlit as st
import pandas as pd
import requests
import time
from utils import create_training_job


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
    elif st.session_state.get("dataset_id") and not st.session_state.get("preview_loaded"):
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
                        st.session_state.preview_data = df_preview
                        st.session_state.preview_loaded = True
                        st.dataframe(df_preview, width="stretch")
            except Exception as e:
                st.warning(f"Could not load preview from API: {e}")
    elif st.session_state.get("preview_loaded") and st.session_state.get("preview_data") is not None:
        st.dataframe(st.session_state.preview_data, width="stretch")

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
        "Select  tabular feature column(s)",
        options=suggested_features,
    )
    st.session_state["feature_cols"] = feature_cols

    # 2) Multi-select modality columns
    modality_cols = st.multiselect(
        "Select modality column(s) (image, text)",
        options=suggested_modalities,
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
        target_col = st.selectbox(
            "Select target column", 
            options=available_targets,
            index=None
        )
        st.session_state["target_col"] = target_col

    # ------- Training configuration (put this BEFORE the nav buttons) -------
    # Defaults
    st.session_state.setdefault("train_pct", 80)
    st.session_state.setdefault("test_pct", 20)
    st.session_state.setdefault("epochs", 4)
    st.session_state.setdefault("learning_rate", 1e-4)
    st.session_state.setdefault("_split_sync_guard", False)

    def _sync_from_train():
        if st.session_state._split_sync_guard:
            return
        st.session_state._split_sync_guard = True
        st.session_state.train_pct = max(1, min(99, int(st.session_state.train_pct)))
        st.session_state.test_pct = 100 - st.session_state.train_pct
        st.session_state._split_sync_guard = False

    def _sync_from_test():
        if st.session_state._split_sync_guard:
            return
        st.session_state._split_sync_guard = True
        st.session_state.test_pct = max(1, min(99, int(st.session_state.test_pct)))
        st.session_state.train_pct = 100 - st.session_state.test_pct
        st.session_state._split_sync_guard = False

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input(
            "Train %",
            min_value=1, max_value=99, step=1,
            key="train_pct",
            on_change=_sync_from_train
        )

    with c2:
        st.number_input(
            "Test %",
            min_value=1, max_value=99, step=1,
            key="test_pct",
            on_change=_sync_from_test
        )

    with c3:
        st.number_input(
            "Epochs",
            min_value=1, max_value=1000, step=1,
            key="epochs"
        )

    with c4:
        st.number_input(
            "Learning rate",
            min_value=1e-6, max_value=1.0, step=1e-5,
            format="%.6f",
            key="learning_rate"
        )

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅️ Back", on_click=on_back, width="stretch")

    with col2:
        def _create_and_go():
            try:
                job_id = create_training_job(api_url)
                st.session_state.training_job_id = job_id
                st.success(f"Training job #{job_id} created.")
                st.session_state.page = "Train & Results"   # go to Page 3
            except Exception as e:
                st.error(f"Failed to create training job: {e}")
        is_next_enabled = (
            target_col is not None and 
            (modality_cols or feature_cols)
        )
        st.button(
            "➡️ Next (create job)",
            on_click=_create_and_go,
            width="stretch",
            disabled=not is_next_enabled
        )
