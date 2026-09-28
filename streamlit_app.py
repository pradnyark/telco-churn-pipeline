"""
streamlit_app.py - Cloud Dashboard for the Telco Churn Data Pipeline

Reads run logs and plots produced by preprocess_eda.py and displays them
as a live dashboard. Deploy this for free on Streamlit Community Cloud
(share.streamlit.io) to get a genuine hosted cloud dashboard with a
public URL -- no AWS required.

Run locally with:
    pip install streamlit pandas glob2 pillow
    streamlit run streamlit_app.py

To deploy to the cloud (free):
    1. Push this file + your output/ folder to a public GitHub repo
    2. Go to https://share.streamlit.io, sign in with GitHub
    3. Click 'New app', select your repo, set main file to streamlit_app.py
    4. Click Deploy -- you'll get a public URL like
       https://yourapp.streamlit.app
"""

import os
import json
import glob
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Telco Churn Pipeline Dashboard", layout="wide")

OUTPUT_DIR = "output"

st.title("📊 Telco Churn Data Pipeline — Cloud Dashboard")
st.caption("Live monitoring of the automated preprocessing + EDA pipeline (runs every 2 minutes)")


def load_run_logs():
    log_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "run_log_*.json")))
    logs = []
    for lf in log_files:
        with open(lf) as f:
            data = json.load(f)
            data["_file"] = os.path.basename(lf)
            logs.append(data)
    return logs


logs = load_run_logs()

if not logs:
    st.warning("No pipeline runs found yet. Run `preprocess_eda.py` at least once, "
               "or make sure this app is pointed at the same `output/` folder.")
else:
    latest = logs[-1]

    # --- Top metrics row ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs Logged", len(logs))
    col2.metric("Rows Processed (latest)", latest.get("input_rows", "N/A"))
    col3.metric("Missing-Value Columns (latest)", len(latest.get("missing_value_columns", {})))
    col4.metric("Last Run Time", latest.get("run_end", "N/A")[:19].replace("T", " "))

    st.divider()

    # --- Run history table ---
    st.subheader("Run History")
    history_df = pd.DataFrame([
        {
            "Run End": l.get("run_end", "")[:19].replace("T", " "),
            "Rows Processed": l.get("input_rows"),
            "Missing Value Columns": len(l.get("missing_value_columns", {})),
        }
        for l in logs
    ])
    st.dataframe(history_df, use_container_width=True)

    # --- Trend chart ---
    if len(logs) > 1:
        st.subheader("Rows Processed Over Time")
        st.line_chart(history_df.set_index("Run End")["Rows Processed"])

    st.divider()

    # --- Latest run's top features ---
    st.subheader("Top Features (latest run)")
    top_features = latest.get("top_features", {})
    if top_features:
        feat_df = pd.DataFrame(list(top_features.items()), columns=["Feature", "Importance"])
        st.bar_chart(feat_df.set_index("Feature"))

    st.divider()

    # --- Plots from the latest run ---
    st.subheader("EDA Visuals (latest run)")
    plot_cols = st.columns(2)
    plot_files = ["correlation_heatmap.png", "feature_importance.png"]
    for i, pf in enumerate(plot_files):
        path = os.path.join(OUTPUT_DIR, pf)
        if os.path.exists(path):
            plot_cols[i % 2].image(Image.open(path), caption=pf, use_column_width=True)

    # Any univariate/bivariate plots found
    extra_plots = glob.glob(os.path.join(OUTPUT_DIR, "univariate_*.png")) + \
                  glob.glob(os.path.join(OUTPUT_DIR, "bivariate_*.png"))
    if extra_plots:
        st.subheader("Univariate / Bivariate Plots")
        cols = st.columns(3)
        for i, p in enumerate(extra_plots):
            cols[i % 3].image(Image.open(p), caption=os.path.basename(p), use_column_width=True)

st.divider()
st.caption(f"Dashboard refreshed at {datetime.utcnow().isoformat()} UTC — reload the page to see the latest run.")
