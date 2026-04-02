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
RENDER_DEPLOY_URL = "https://summative-assignment-mlop-9yqj.onrender.com"
if "onrender.com" in os.environ.get("RENDER_EXTERNAL_HOSTNAME", "") or "onrender.com" in os.environ.get("RENDER", ""):
    API_URL = RENDER_DEPLOY_URL
else:
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Initialize Session State
if 'data_ready' not in st.session_state:
    st.session_state['data_ready'] = False

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
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Smart Waste Classifier Pro")
st.markdown("---")

# Sidebar - System Monitoring
st.sidebar.title("System Monitor")
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    status_text = "Status: ONLINE" if health['status'] == 'up' else "Status: DEGRADED"
    st.sidebar.markdown(status_text)
    st.sidebar.metric("System Uptime", health['uptime'])
    st.sidebar.info(f"Active Model Version: {health['model_version']}")
    
    # Retrain Status Polling
    retrain_status = requests.get(f"{API_URL}/retrain/status", timeout=2).json()
    if retrain_status['is_retraining']:
        st.sidebar.warning("Retraining in progress...")
        st.sidebar.progress(0.5)
    else:
        st.sidebar.success(f"Last Event: {retrain_status['last_status']}")
except Exception as e:
    st.sidebar.error("Connection Error: API is unreachable")
    st.sidebar.info("Please ensure the API is running at http://localhost:8000")

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
        st.subheader("Neural Inference Results")
        if uploaded_file and st.button("Execute Prediction", use_container_width=True):
            with st.spinner('Running Model Inference...'):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(f"{API_URL}/predict", files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Results display
                        if data.get('is_uncertain'):
                            st.warning("Low Confidence Warning: The system is unable to classify with high certainty.")
                            st.error(f"Suggested Category: {data['class_name']}")
                        else:
                            st.success(f"Classification Result: {data['class_name']}")
                        
                        st.write(f"Confidence Level: {data['confidence']:.2%}")
                        st.progress(data['confidence'])
                        
                        # Show Heatmap
                        if data.get('heatmap_base64'):
                            st.subheader("Visual Explanation (Grad-CAM)")
                            st.markdown("The heatmap below identifies the specific features the model utilized for this classification.")
                            heatmap_bytes = base64.b64decode(data['heatmap_base64'])
                            st.image(heatmap_bytes, caption="Attention Map", use_container_width=True)

                        # Probability Distribution
                        scores_df = pd.DataFrame(data['all_scores'].items(), columns=['Category', 'Probability'])
                        fig = px.bar(scores_df, x='Probability', y='Category', orientation='h', 
                                     title="Class Probability Distribution",
                                     color='Probability', color_continuous_scale='Greens')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Inference process failed: {e}")

# 2. Data Insights Tab
with tab_viz:
    st.header("Dataset and Model Performance Metrics")
    
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
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
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
                                     title="Model Accuracy Convergence")
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
    st.header("Model Lifecycle and Retraining Pipeline")
    st.markdown("Manage bulk data uploads and trigger model retraining events.")
    
    with st.expander("Step 1: Data Acquisition"):
        new_zip = st.file_uploader("Upload labeled dataset (.zip format)", type="zip")
        if new_zip:
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
        retrain_button = st.button("Initiate Retraining Process")
        if retrain_button:
            if not st.session_state.get('data_ready'):
                st.warning("No new dataset staged. Retraining will use the current dataset.")
            try:
                retrain_resp = requests.post(f"{API_URL}/retrain")
                if retrain_resp.status_code == 200:
                    st.toast("Retraining Pipeline Initiated")
                    st.info("Model fine-tuning is running in the background.")
                else:
                    st.error("Failed to initiate retraining.")
            except Exception as e:
                st.error(f"Network error during training trigger: {e}")

    with st.expander("Step 3: Model Registry and Database Records"):
        try:
            r_status = requests.get(f"{API_URL}/retrain/status", timeout=2).json()
            registry = r_status.get('registry', {})
            
            st.subheader("Data Upload History (SQLite Records)")
            if registry.get('uploads'):
                up_df = pd.DataFrame(registry['uploads'])
                st.table(up_df[['timestamp', 'filename', 'file_size_kb']])
            else:
                st.info("No data uploads recorded.")

            st.subheader("Model Training Lineage")
            if registry.get('history'):
                hist_df = pd.DataFrame(registry['history'])
                st.table(hist_df[['timestamp', 'model_path', 'accuracy', 'status']])
            else:
                st.info("No training history recorded.")
        except Exception as e:
            st.info(f"Registry retrieval error: {e}")
