import streamlit as st  # Trigger redeploy
import requests
import os
import time
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Initialize Session State
if 'data_ready' not in st.session_state:
    st.session_state['data_ready'] = False
if 'prev_retraining' not in st.session_state:
    st.session_state['prev_retraining'] = False

st.set_page_config(
    page_title="Smart Waste Classifier Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    /* Professional Dark Sidebar for high contrast */
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
        background-color: #111827 !important;
    }
    /* Ensure all text elements in sidebar are visible */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption {
        color: #f9fafb !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e1e4e8;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .log-text {
        font-family: 'Courier New', Courier, monospace;
        font-size: 12px;
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Smart Waste Classifier Pro")
st.markdown("---")

# Sidebar - System Monitoring

# Professional Logo Placement
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png", use_container_width=True)
else:
    # Vertical spacer to maintain alignment if logo is missing
    st.sidebar.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

st.sidebar.title("System Control Center")
is_retraining = False  # Global flag for UI state

try:
    # Fetch global info from root
    root_info = requests.get(API_URL, timeout=2).json()
    health = requests.get(f"{API_URL}/health", timeout=2).json()

    if health['status'] == 'up':
        st.sidebar.success("System Online")
    else:
        st.sidebar.error("System Degraded")

    st.sidebar.metric("Uptime", health['uptime'])
    st.sidebar.caption(f"Model ID: `{health['model_version']}`")

    # Retrain Status Polling
    retrain_status = requests.get(f"{API_URL}/retrain/status", timeout=2).json()
    is_retraining = retrain_status.get('is_retraining', False)
    last_status = retrain_status.get('last_status', 'none')

    # Detect transition from retraining to finished
    if not is_retraining and st.session_state.get('prev_retraining', False):
        if last_status == "success":
            st.toast("Retraining Complete! The new Champion model is now serving live requests.")
            st.balloons()

    st.session_state['prev_retraining'] = is_retraining

    if is_retraining:
        st.sidebar.warning("Retraining in progress...")
        st.sidebar.progress(0.65)
        # Auto-refresh UI while retraining is active
        time.sleep(5)
        st.rerun()
    else:
        st.sidebar.info(f"Last Status: {last_status}")

    with st.sidebar.expander("Live System Logs", expanded=True): # Expanded by default for visibility
        log_levels = ["ALL", "INFO", "WARNING", "ERROR", "CRITICAL"]
        selected_level = st.selectbox("Filter by Level", log_levels, key="log_level_filter")

        logs = root_info.get('recent_logs', [])

        filtered_logs = []
        if selected_level == "ALL":
            filtered_logs = logs
        else:
            filtered_logs = [log for log in logs if log.get('level') == selected_level]

        formatted_log_lines = [log.get('full_formatted', f"{log.get('timestamp', '')} {log.get('level', '')}: {log.get('message', '')}") for log in filtered_logs]
        log_content = "\n".join(formatted_log_lines) if formatted_log_lines else "No logs to display for selected level."
        st.markdown(f'<div class="log-text">{log_content}</div>', unsafe_allow_html=True) # Use full_formatted

except Exception as e:
    st.sidebar.error("Connection Error: API is unreachable")
    st.sidebar.info("Please ensure the API is running at http://localhost:8000")

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='font-size: 0.85em; color: #9ca3af;'>
        Software Version: 1.0.0-stable<br>
        <a href='#' style='color: #60a5fa; text-decoration: none;'>Technical Documentation</a>
    </div>
""", unsafe_allow_html=True)

# Main Tabs
tab_pred, tab_viz, tab_mlops = st.tabs(["Intelligent Prediction", "Data Insights", "MLOps Pipeline"])

# 1. Prediction Tab
with tab_pred:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Image Acquisition")
        uploaded_file = st.file_uploader("Upload waste image for analysis...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Input Stream", use_container_width=True)

    with col2:
        st.subheader("Neural Inference Engine")
        if uploaded_file and st.button("Execute Prediction", use_container_width=True):
            with st.spinner('Running Model Inference...'):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(f"{API_URL}/predict", files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Card-style results
                        with st.container():
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            if data.get('is_uncertain'):
                                st.warning("Uncertainty Detected")
                                st.subheader(f"Suggested: {data['class_name']}")
                            else:
                                st.success(f"Prediction: {data['class_name']}")
                            
                            st.write(f"Confidence: **{data['confidence']:.2%}** | Latency: **{data['latency']:.3f}s**")
                            st.progress(data['confidence'])
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show Heatmap
                        if data.get('heatmap_base64'):
                            st.subheader("Interpretability (Grad-CAM)")
                            st.caption("Model attention map: Red areas indicate features that influenced the decision most.")
                            heatmap_bytes = base64.b64decode(data['heatmap_base64'])
                            st.image(heatmap_bytes, caption="Attention Map", use_container_width=True)

                        # Probability Distribution
                        scores_df = pd.DataFrame(data['all_scores'].items(), columns=['Category', 'Probability'])
                        fig = px.bar(scores_df, x='Probability', y='Category', orientation='h', 
                                     title="Probability Distribution",
                                     color='Probability', color_continuous_scale='Greens')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Inference process failed: {e}")

# 2. Data Insights Tab
with tab_viz:
    st.header("Analytical Insights")
    
    try:
        stats_resp = requests.get(f"{API_URL}/stats", timeout=5)
        if stats_resp.status_code == 200:
            stats_data = stats_resp.json()
            df_dataset = pd.DataFrame(stats_data['dataset'])
            df_history = pd.DataFrame(stats_data['history'])
            
            st.subheader("Dataset Statistics")
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                if not df_dataset.empty:
                    fig_bar = px.bar(df_dataset, x="class", y="count", color="split", barmode="group",
                                    title="Class Distribution by Split",
                                    color_discrete_sequence=px.colors.qualitative.G10)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Dataset statistics are currently unavailable.")

            with col_dist2:
                if not df_dataset.empty:
                    df_total = df_dataset.groupby('class')['count'].sum().reset_index()
                    fig_pie = px.pie(df_total, values='count', names='class', hole=0.5,
                                    title="Overall Category Composition")
                    st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Training and Production Monitoring")
            col_h1, col_h2 = st.columns(2)
            
            with col_h1:
                if not df_history.empty:
                    fig_acc = px.line(df_history, x=df_history.index, y=['accuracy', 'val_accuracy'],
                                     title="Training Convergence",
                                     labels={"index": "Epoch", "value": "Accuracy"})
                    st.plotly_chart(fig_acc, use_container_width=True)
                else:
                    st.info("No training history available.")

            with col_h2:
                st.markdown("### Inference Latency Tracking")
                history_resp = requests.get(f"{API_URL}/history", timeout=2).json()
                if history_resp:
                    df_history_live = pd.DataFrame(history_resp)
                    fig_latency = px.area(df_history_live, x="timestamp", y="latency", 
                                         title="Real-time Inference Speed (Seconds)",
                                         color_discrete_sequence=['#FF5733'])
                    st.plotly_chart(fig_latency, use_container_width=True)
                else:
                    st.info("No live inference history detected.")
        else:
            st.error("Failed to load statistics from API.")

    except Exception as e:
        st.error(f"Failed to retrieve analytical data: {e}")

# 3. MLOps Tab
with tab_mlops:
    st.header("MLOps Lifecycle Management")
    st.markdown("Automate data ingestion and model promotion.")

    # Prominent UI Loader for Retraining
    if is_retraining:
        st.info("**Model Fine-Tuning in Progress**")
        st.status("The engine is currently training on the latest dataset and evaluating performance against the Champion model. "
                  "The dashboard will refresh automatically upon completion.", state="running")

    with st.expander("Step 1: Data Ingestion"):
        new_zip = st.file_uploader("Staging Area: Upload dataset (.zip)", type="zip")
        if new_zip and not is_retraining:
            if st.button("Upload and Stage Data"):
                with st.spinner("Uploading dataset to server..."):
                    try:
                        up_resp = requests.post(f"{API_URL}/upload-data", files={"file": (new_zip.name, new_zip.getvalue())})
                        if up_resp.status_code == 200:
                            st.success("Dataset successfully staged and recorded in database.")
                            st.session_state['data_ready'] = True
                        else:
                            st.error(f"Upload process failed: {up_resp.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Network error during upload: {e}")

    with st.expander("Step 2: Training Execution"):
        retrain_button = st.button("Trigger Retraining Pipeline", disabled=is_retraining)
        if retrain_button:
            if not st.session_state.get('data_ready'):
                st.warning("No new dataset staged. Retraining will use the current dataset.")
            try:
                retrain_resp = requests.post(f"{API_URL}/retrain")
                if retrain_resp.status_code == 200:
                    st.toast("Retraining Pipeline Active")
                    st.info("Model fine-tuning is running in the background.")
                else:
                    st.error("Failed to initiate retraining.")
            except Exception as e:
                st.error(f"Network error during training trigger: {e}")

    with st.expander("Step 3: Model Registry Audit"):
        try:
            r_status = requests.get(f"{API_URL}/retrain/status", timeout=2).json()
            registry = r_status.get('registry', {})
            
            st.subheader("Data Upload History (SQLite Records)")
            if registry.get('uploads'):
                up_df = pd.DataFrame(registry['uploads'])
                st.table(up_df[['timestamp', 'filename', 'file_size_kb']])
            else:
                st.info("No data uploads recorded.")

            st.subheader("Model Comparison: Champion vs. Challengers")
            if registry.get('history'):
                hist_df = pd.DataFrame(registry['history'])
                champion_data = registry.get('champion', {})
                champ_path = champion_data.get('model_path')

                # Identify roles and prepare comparison data
                hist_df['Role'] = hist_df['model_path'].apply(
                    lambda x: "Champion" if x == champ_path else "Challenger"
                )
                
                # Display comparison table with specialized column formatting
                st.dataframe(
                    hist_df[['Role', 'timestamp', 'accuracy', 'loss', 'status']],
                    column_config={
                        "accuracy": st.column_config.NumberColumn("Accuracy Score", format="%.2%"),
                        "loss": st.column_config.NumberColumn("Log Loss", format="%.4f"),
                        "timestamp": "Training Date",
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No training history recorded.")
        except Exception as e:
            st.info(f"Registry retrieval error: {e}")
