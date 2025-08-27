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
