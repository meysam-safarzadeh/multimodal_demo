# upload_page.py
import requests
import streamlit as st

@st.cache_data(ttl=30)
def fetch_datasets(api_url: str):
    r = requests.get(f"{api_url}/datasets/", timeout=30)
    r.raise_for_status()
    return r.json()

def render_upload_page(api_base: str, go):
    st.title("📂 Step 1: Choose a Dataset")

    # init state
    st.session_state.setdefault("dataset_name", "demo-dataset")
    st.session_state.setdefault("csv_blob", None)
    st.session_state.setdefault("data_files", [])
    st.session_state.setdefault("folder_files_count", 0)
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("dataset_id", None)
    st.session_state.setdefault("detect_result", None)

    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.caption("Pick an existing dataset from the backend, or upload a new one.")
    with top_right:
        if st.button("🔄 Refresh"):
            fetch_datasets.clear()
            st.rerun()

    # load datasets
    datasets = []
    try:
        datasets = fetch_datasets(api_base) or []
    except Exception as e:
        st.warning(f"Couldn’t load datasets: {e}")

    # build options
    options = ["➕ Upload new dataset"]
    label_to_item = {}
    for item in datasets:
        ds_id = item.get("id")
        name = item.get("name", f"dataset-{ds_id}")
        created = (item.get("created_at") or "")[:19].replace("T", " ")
        label = f"[{ds_id}] {name} • {created}"
        options.append(label)
        label_to_item[label] = item

    choice = st.selectbox("Select existing dataset or upload new:", options, index=0)

    # existing dataset branch
    if choice != "➕ Upload new dataset":
        selected = label_to_item[choice]
        st.session_state.dataset = selected
        st.session_state.dataset_id = selected.get("id")

        # normalize detection summary whether nested (selected["detection"]) or merged
        det = selected.get("detection") or selected
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

        st.success(f"Selected dataset #{st.session_state.dataset_id}: {selected.get('name')}")
        st.write(f"CSV: {selected.get('csv_s3_uri') or '—'}")
        st.write(f"Data: {selected.get('data_s3_uri') or '—'}")

        c1, c2 = st.columns(2)
        with c1:
            st.button("⬅️ Back", on_click=lambda: go("Welcome"), use_container_width=True)
        with c2:
            st.button("➡️ Next", on_click=lambda: go("Training Configuration"), use_container_width=True)
        return

    # upload-new branch
    st.divider()
    st.subheader("Upload new dataset")

    st.session_state.dataset_name = st.text_input(
        "Dataset name", value=st.session_state.dataset_name
    )

    csv_file = st.file_uploader("Upload metadata CSV", type=["csv"])
    if csv_file is not None:
        st.session_state.csv_blob = csv_file.getvalue()
        st.success(f"CSV uploaded: {csv_file.name}")
        # reset any prior selection/results
        st.session_state.dataset = None
        st.session_state.dataset_id = None
        st.session_state.detect_result = None

    folder_files = st.file_uploader(
        "Upload data folder (multiple files)", type=None, accept_multiple_files=True
    )
    if folder_files:
        st.session_state.data_files = folder_files
        st.session_state.folder_files_count = len(folder_files)
        st.success(f"{len(folder_files)} files uploaded")

    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅️ Back", on_click=lambda: go("Welcome"), use_container_width=True)
    with c2:
        ready = bool(st.session_state.csv_blob) and st.session_state.folder_files_count > 0
        st.button("➡️ Next", disabled=not ready, on_click=lambda: go("Training Configuration"), use_container_width=True)
