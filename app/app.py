import streamlit as st
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
API_URL = os.getenv("API_URL", "http://api:8000")

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
    health = requests.get(f"{API_URL}/health").json()
    status_text = "Status: ONLINE" if health['status'] == 'up' else "Status: DEGRADED"
    st.sidebar.markdown(status_text)
    st.sidebar.metric("System Uptime", health['uptime'])
    st.sidebar.info(f"Active Model Version: {health['model_version']}")
    
    # Retrain Status Polling
    retrain_status = requests.get(f"{API_URL}/retrain/status").json()
    if retrain_status['is_retraining']:
        st.sidebar.warning("Retraining in progress...")
        st.sidebar.progress(0.5)
    else:
        st.sidebar.success(f"Last Retrain Event: {retrain_status['last_status']}")
except Exception as e:
    st.sidebar.error("Connection Error: API is unreachable")

# Main Tabs
tab_pred, tab_viz, tab_mlops = st.tabs(["Intelligent Prediction", "Data Insights", "MLOps Pipeline"])

# 1. Prediction Tab
with tab_pred:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Image Acquisition")
        uploaded_file = st.file_uploader("Upload waste image for analysis...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Input Stream", use_column_width=True)

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
                        st.subheader("Visual Explanation (Grad-CAM)")
                        st.markdown("The heatmap below identifies the specific features the model utilized for this classification.")
                        heatmap_bytes = base64.b64decode(data['heatmap_base64'])
                        st.image(heatmap_bytes, caption="Attention Map", use_column_width=True)

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
        stats_data = requests.get(f"{API_URL}/stats").json()
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
            history_resp = requests.get(f"{API_URL}/history").json()
            if history_resp:
                df_history_live = pd.DataFrame(history_resp)
                fig_latency = px.area(df_history_live, x="timestamp", y="latency", 
                                     title="Real-time Inference Speed (Seconds)",
                                     color_discrete_sequence=['#FF5733'])
                st.plotly_chart(fig_latency, use_container_width=True)
            else:
                st.info("No inference history detected.")

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
                    up_resp = requests.post(f"{API_URL}/upload-data", files={"file": (new_zip.name, new_zip.getvalue())})
                    if up_resp.status_code == 200:
                        st.success("Dataset successfully staged for retraining.")
                        st.session_state['data_ready'] = True
                    else:
                        st.error("Upload process failed.")

    with st.expander("Step 2: Training Execution"):
        if st.session_state.get('data_ready'):
            if st.button("Initiate Retraining Process"):
                retrain_resp = requests.post(f"{API_URL}/retrain")
                if retrain_resp.status_code == 200:
                    st.toast("Retraining Pipeline Initiated")
                    st.info("Model fine-tuning is running in the background. Monitor the sidebar for status updates.")
                else:
                    st.error("Failed to initiate retraining.")
        else:
            st.write("Please stage a dataset before initiating training.")

    with st.expander("Step 3: Model Registry and Validation"):
        try:
            r_status = requests.get(f"{API_URL}/retrain/status").json()
            registry = r_status.get('registry', {})
            
            if registry.get('champion'):
                st.subheader("Current Production Model (Champion)")
                champ = registry['champion']
                st.json({
                    "Model Path": champ['path'],
                    "Accuracy": champ['metrics'].get('accuracy'),
                    "Validation Date": time.ctime(champ['timestamp'])
                })
            
            if registry.get('history'):
                st.subheader("Model Lineage and Audit Trail")
                hist_df = pd.DataFrame([
                    {
                        "Version": h['path'].split('/')[-1],
                        "Accuracy": h['metrics'].get('accuracy'),
                        "Timestamp": time.ctime(h['timestamp'])
                    } for h in registry['history']
                ])
                st.table(hist_df)
        except:
            st.info("Model registry is currently empty.")
