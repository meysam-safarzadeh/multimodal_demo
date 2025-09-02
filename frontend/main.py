import streamlit as st
from train_page import render_train_page

st.set_page_config(page_title="Multimodal Trainer", page_icon="🤖", layout="centered")

# Configure API URL (override via .streamlit/secrets.toml if needed)
API_URL = "http://localhost:8000"

# --- Session state init ---
if "page" not in st.session_state:
    st.session_state.page = "Welcome"
if "csv_blob" not in st.session_state:
    st.session_state.csv_blob = None
if "folder_files_count" not in st.session_state:
    st.session_state.folder_files_count = 0

def go(page: str):
    st.session_state.page = page

# --- PAGE 0: Welcome ---
if st.session_state.page == "Welcome":
    st.title("🤖 Welcome to the Multimodal Trainer Demo")
    st.write("This wizard will take you from data upload to training configuration.")
    st.button("🚀 Start", on_click=lambda: go("Upload Data"))

# --- PAGE 1: Upload Data ---
elif st.session_state.page == "Dataset Selection":
    st.title("📂 Step 1: Upload Your Data")

    # optional dataset name
    st.session_state.setdefault("dataset_name", "demo-dataset")
    st.session_state.setdefault("csv_blob", None)
    st.session_state.setdefault("data_files", [])
    st.session_state.setdefault("folder_files_count", 0)

    st.session_state.dataset_name = st.text_input(
        "Dataset name", value=st.session_state.dataset_name
    )

    # CSV
    csv_file = st.file_uploader("Upload metadata CSV", type=["csv"])
    if csv_file is not None:
        st.session_state.csv_blob = csv_file.getvalue()
        st.success(f"CSV uploaded: {csv_file.name}")
        # reset any previous results when user re-uploads
        st.session_state.pop("dataset", None)
        st.session_state.pop("dataset_id", None)
        st.session_state.pop("detect_result", None)

    # Folder files (multiple)
    folder_files = st.file_uploader(
        "Upload data folder (multiple files)", type=None, accept_multiple_files=True
    )
    if folder_files:
        # store the UploadedFile objects for page 2 (we'll read bytes there)
        st.session_state.data_files = folder_files
        st.session_state.folder_files_count = len(folder_files)
        st.success(f"{len(folder_files)} files uploaded")

    # Nav
    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅️ Back", on_click=lambda: go("Welcome"))
    with col2:
        ready = bool(st.session_state.csv_blob) and st.session_state.folder_files_count > 0
        st.button("➡️ Next", disabled=not ready, on_click=lambda: go("Train Model"))


# --- PAGE 2: Train Model ---
elif st.session_state.page == "Train Model":
    render_train_page(
        api_url=API_URL,
        csv_blob=st.session_state.csv_blob,
        on_back=lambda: go("Upload Data"),
        on_home=lambda: go("Welcome"),
    )
