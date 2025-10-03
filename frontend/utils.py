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


def render_training_curves(metrics: dict):
    """
    Render accuracy and loss curves from metrics['logs'] if present.
    Supports several shapes:
      logs = {"curve": [ {epoch, acc, val_acc, loss, val_loss}, ... ]}
      logs = {"history": {"epoch":[...], "acc":[...], "val_acc":[...], ...}}
      logs = [ {epoch, ...}, ... ]
      logs = {"acc":[...], "val_acc":[...], ...}
    """
    logs = (metrics or {}).get("logs")
    if not logs:
        st.caption("No training logs available.")
        return

    df = None

    # Case A: {"curve": [ {...}, {...} ]}
    if isinstance(logs, dict) and isinstance(logs.get("curve"), list):
        df = pd.DataFrame(logs["curve"])

    # Case B: {"history": {"acc":[...], ...}}
    elif isinstance(logs, dict) and isinstance(logs.get("history"), dict):
        hist = logs["history"]
        # figure out length
        lens = [len(v) for v in hist.values() if isinstance(v, list)]
        if lens:
            L = max(lens)
            epochs = hist.get("epoch") or list(range(1, L + 1))
            df = pd.DataFrame({"epoch": epochs})
            for k, v in hist.items():
                if k == "epoch" or not isinstance(v, list):
                    continue
                df[k] = v[: len(df)]

    # Case C: list of dicts
    elif isinstance(logs, list):
        df = pd.DataFrame(logs)

    # Case D: dict of lists
    elif isinstance(logs, dict) and all(isinstance(v, list) for v in logs.values()):
        lens = [len(v) for v in logs.values()]
        L = max(lens) if lens else 0
        df = pd.DataFrame({"epoch": list(range(1, L + 1))})
        for k, v in logs.items():
            df[k] = v[: len(df)]

    # Nothing workable
    if df is None or df.empty:
        st.caption("No curve-like data found in logs.")
        return

    # Normalize column name lookups (case-insensitive)
    norm = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in norm:
                return norm[c]
        return None

    epoch = pick("epoch")
    if epoch is None:
        df.insert(0, "epoch", range(1, len(df) + 1))
        epoch = "epoch"

    acc = pick("acc", "accuracy", "train_acc", "train_accuracy")
    val_acc = pick("val_acc", "val_accuracy")
    loss = pick("loss", "train_loss")
    val_loss = pick("val_loss", "validation_loss")

    # Build plots
    df_plot = df.set_index(epoch)

    # Accuracy chart
    acc_cols = [c for c in [acc, val_acc] if c]
    if acc_cols:
        st.markdown("**Accuracy**")
        st.line_chart(df_plot[acc_cols], width="stretch", color=["#FF5733", "#3498DB"])

    # Loss chart
    loss_cols = [c for c in [loss, val_loss] if c]
    if loss_cols:
        st.markdown("**Loss**")
        st.line_chart(df_plot[loss_cols], width="stretch", color=["#FF5733", "#3498DB"])

    if not acc_cols and not loss_cols:
        st.caption("Logs found, but no recognizable acc/loss keys to plot.")


def _human_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{n} B"


def render_artifacts_section(api_url: str, job_id: int, auth_header: dict | None = None):
    """
    Renders artifact downloads using the presigned response:
    [
      {
        "key": "id2label",
        "s3_uri": "s3://bucket/artifacts/<job_id>/id2label.json",
        "filename": "id2label.json",
        "url": "https://...presigned...",
        "content_length": 50,
        "content_type": "application/json",
        "expires_in": 3600
      },
      ...
    ]
    """
    st.subheader("📦 Artifacts")

    try:
        r = requests.get(
            f"{api_url}/training_jobs/{job_id}/artifacts/presign/",
            headers=auth_header or {},
            timeout=30,
        )
        r.raise_for_status()
        items = r.json()
        if isinstance(items, dict):  # API may return a single object when ?key=...
            items = [items]
        if not items:
            st.info("No artifacts yet. They’ll appear here after training finishes.")
            return

        # Header with quick refresh (any button click reruns Streamlit)
        cols_top = st.columns([2, 2, 2, 2])
        cols_top[0].markdown("**Key**")
        cols_top[1].markdown("**Filename**")
        cols_top[2].markdown("**Size**")
        cols_top[3].button("🔄 Refresh links", width="stretch")

        for art in items:
            key = art.get("key", "artifact")
            filename = art.get("filename") or key
            url = art.get("url")
            size = art.get("content_length")

            c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
            with c1:
                st.markdown(f"**{key}**")
            with c2:
                st.caption(f"📄 {filename}")
            with c3:
                st.metric("Size", _human_bytes(size))
            with c4:
                if url:
                    st.link_button(f"⬇️ Download {filename}", url, width="stretch")
                else:
                    st.button("⬇️ Download", disabled=True, width="stretch")

        # Footer note about expiry
        # (use the first item's expires_in as a hint; presigned TTLs are per-call)
        first_exp = items[0].get("expires_in")
        if first_exp:
            minutes = max(1, int(first_exp // 60))
            st.caption(f"Links expire in ~{minutes} minute(s). Click **Refresh links** to regenerate.")

    except Exception as e:
        st.warning(f"No artifacts yet. They’ll appear here after training finishes.")
