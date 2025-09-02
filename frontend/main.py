import streamlit as st
from upload_page import render_upload_page
from config_page import render_config_page

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
    st.button("🚀 Start", on_click=lambda: go("Dataset Selection"))

# --- PAGE 1: Upload Data ---
elif st.session_state.page == "Dataset Selection":
    render_upload_page(API_URL, go)

# --- PAGE 2: Train Model ---
elif st.session_state.page == "Train Model":
    render_config_page(
        api_url=API_URL,
        csv_blob=st.session_state.csv_blob,
        on_back=lambda: go("Dataset Selection"),
        on_home=lambda: go("Welcome"),
    )
