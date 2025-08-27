import streamlit as st
from metadata_detector import MetadataDetector
import time
import tempfile

def render_train_page(csv_file):
    st.title("⚙️ Step 2: Train Model")

    if csv_file is None:
        st.warning("⚠️ Please upload a CSV file on the previous page.")
        return

    # Save uploaded file to a temporary path so MetadataDetector can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(csv_file.getvalue())
        tmp_path = tmp.name

    detector = MetadataDetector(tmp_path)
    summary = detector.detect()

    # --- CSV preview ---
    st.subheader("📄 CSV Preview")
    st.dataframe(detector.df.head(10))

    # --- Metadata summary ---
    st.subheader("🔍 Detected Metadata")
    st.write(f"**Rows:** {summary['n_rows']}, **Columns:** {summary['n_columns']}")
    st.write(f"**Modalities detected:** {', '.join(summary['modalities']) if summary['modalities'] else 'None'}")

    st.markdown("**Column Types:**")
    st.json(summary["column_types"])

    st.markdown("**Missing Data per Column:**")
    st.json(summary["missing_data"])

    # --- Dropdowns ---
    st.subheader("⚙️ Configure Training")

    # 1) Multi-select feature columns
    feature_cols = st.multiselect("Select feature columns", summary["feature_columns"])

    # 2) Multi-select modality columns (unchanged)
    modality_cols = st.multiselect("Select modality columns", summary["modality_columns"])

    # 3) Target column excludes any selected feature columns
    available_targets = [c for c in summary["target_columns_categorical"] if c not in feature_cols]

    if not available_targets:
        st.warning("You have selected all categorical columns as features. Please deselect some to choose a target.")
        target_col = None
    else:
        # (Optional) keep prior selection if still valid
        prev = st.session_state.get("target_col")
        default_index = available_targets.index(prev) if prev in available_targets else 0
        target_col = st.selectbox("Select target column", available_targets, index=default_index)
        st.session_state["target_col"] = target_col

    if st.button("Start Training", disabled=(target_col is None)):
        with st.spinner("Training in progress..."):
            time.sleep(2)  # Placeholder for API call
            st.success("✅ Training complete!")
            st.write("Dummy metrics: Accuracy = 0.92, Loss = 0.13")
            st.write(f"Features: {feature_cols}")
            st.write(f"Modalities: {modality_cols}")
            st.write(f"Target: {target_col}")
            st.download_button("Download Model", data=b"dummy-model", file_name="model.pt")

    st.button("⬅️ Back", on_click=lambda: st.session_state.update(page="Upload Data"))
