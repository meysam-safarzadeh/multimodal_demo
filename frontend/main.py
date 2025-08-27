import streamlit as st
from train_page import render_train_page

st.set_page_config(page_title="Multimodal Trainer", page_icon="🤖", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "Welcome"
if "csv_file" not in st.session_state:
    st.session_state.csv_file = None

# --- PAGE 0: Welcome ---
if st.session_state.page == "Welcome":
    st.title("🤖 Welcome to the Multimodal Trainer Demo")
    st.write("This demo walks you through uploading data and training a model.")
    st.button("🚀 Start", on_click=lambda: st.session_state.update(page="Upload Data"))

# --- PAGE 1: Upload Data ---
elif st.session_state.page == "Upload Data":
    st.title("📂 Step 1: Upload Your Data")

    csv_file = st.file_uploader("Upload metadata CSV", type=["csv"])
    if csv_file is not None:
        st.session_state.csv_file = csv_file
        st.success(f"CSV uploaded: {csv_file.name}")

    folder_files = st.file_uploader(
        "Upload data folder (multiple files)",
        type=None,
        accept_multiple_files=True
    )
    if folder_files:
        st.success(f"{len(folder_files)} files uploaded")

    if csv_file and folder_files:
        st.button("➡️ Next", on_click=lambda: st.session_state.update(page="Train Model"))
    st.button("⬅️ Back", on_click=lambda: st.session_state.update(page="Welcome"))

# --- PAGE 2: Train Model ---
elif st.session_state.page == "Train Model":
    render_train_page(st.session_state.csv_file)
