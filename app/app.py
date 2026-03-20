import streamlit as st
import requests
import os
import time
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Configuration
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="Smart Waste Classifier Pro",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
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

st.title("♻️ Smart Waste Classifier Pro")
st.markdown("---")

# Sidebar - System Monitoring
st.sidebar.title("🛡️ System Monitor")
try:
    health = requests.get(f"{API_URL}/health").json()
    status_color = "green" if health['status'] == 'up' else "orange"
    st.sidebar.markdown(f"Status: :{status_color}[{health['status'].upper()}]")
    st.sidebar.metric("System Uptime", health['uptime'])
    st.sidebar.info(f"Active Model: {health['model_version']}")
    
    # Retrain Status Polling
    retrain_status = requests.get(f"{API_URL}/retrain/status").json()
    if retrain_status['is_retraining']:
        st.sidebar.warning("⚙️ Retraining in progress...")
        st.sidebar.progress(0.5) # Indeterminate
    else:
        st.sidebar.success(f"Last Retrain: {retrain_status['last_status']}")
except Exception as e:
    st.sidebar.error("Could not connect to API")

# Main Tabs
tab_pred, tab_viz, tab_mlops = st.tabs(["🔍 Intelligent Prediction", "📊 Advanced Analytics", "🏗️ MLOps Pipeline"])

# 1. Prediction Tab
with tab_pred:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Image Acquisition")
        uploaded_file = st.file_uploader("Drop waste image here...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Input Stream", use_column_width=True)

    with col2:
        st.subheader("🧠 Model Inference")
        if uploaded_file and st.button("Run Analysis", use_container_width=True):
            with st.spinner('Executing Neural Inference...'):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(f"{API_URL}/predict", files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Results display
                        st.success(f"Classification: **{data['class_name']}**")
                        st.progress(data['confidence'])
                        
                        # Show Heatmap
                        st.subheader("🧐 Why did the model think this?")
                        st.markdown("The **Grad-CAM Heatmap** below shows which areas the neural network focused on (Red/Yellow = High Focus).")
                        heatmap_bytes = base64.b64decode(data['heatmap_base64'])
                        st.image(heatmap_bytes, caption="Model Attention Map", use_column_width=True)

                        # Probability Distribution
                        scores_df = pd.DataFrame(data['all_scores'].items(), columns=['Category', 'Probability'])
                        fig = px.bar(scores_df, x='Probability', y='Category', orientation='h', 
                                     title="Class Probability Distribution",
                                     color='Probability', color_continuous_scale='Greens')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Error: {resp.json()['detail']}")
                except Exception as e:
                    st.error(f"Inference failed: {e}")

# 2. Visualizations Tab
with tab_viz:
    st.header("📈 Data & Model Intelligence Dashboard")
    
    try:
        stats_data = requests.get(f"{API_URL}/stats").json()
        df_dataset = pd.DataFrame(stats_data['dataset'])
        df_history = pd.DataFrame(stats_data['history'])
        
        # Row 1: Dataset Exploration
        st.subheader("📁 Dataset Statistics")
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            if not df_dataset.empty:
                # Class distribution by split
                fig_bar = px.bar(df_dataset, x="class", y="count", color="split", barmode="group",
                                title="Class Distribution (Train vs Test Set)",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No dataset stats found. Please download data.")

        with col_dist2:
            if not df_dataset.empty:
                # Total pie distribution
                df_total = df_dataset.groupby('class')['count'].sum().reset_index()
                fig_pie = px.pie(df_total, values='count', names='class', hole=0.5,
                                title="Overall Category Composition",
                                color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_pie, use_container_width=True)

        # Row 2: Training Performance
        st.subheader("🎯 Training & Production Monitoring")
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            if not df_history.empty:
                fig_acc = px.line(df_history, x=df_history.index, y=['accuracy', 'val_accuracy'],
                                 title="Generalization: Accuracy over Epochs")
                st.plotly_chart(fig_acc, use_container_width=True)
            else:
                st.info("No training history yet.")

        with col_h2:
            st.markdown("### 📡 Live Production Latency")
            history_resp = requests.get(f"{API_URL}/history").json()
            if history_resp:
                df_history_live = pd.DataFrame(history_resp)
                fig_latency = px.area(df_history_live, x="timestamp", y="latency", 
                                     title="Real-time Inference Speed (Seconds)",
                                     color_discrete_sequence=['#FF5733'])
                st.plotly_chart(fig_latency, use_container_width=True)
            else:
                st.info("No live prediction traffic detected yet.")

        # Row 3: Feature Interpretability
        st.subheader("🕵️ Advanced Feature Interpretation")
        feature_col1, feature_col2 = st.columns([1, 2])
        with feature_col1:
            selected_feature = st.radio("Focus Area", ["Micro-Texture", "Geometric Reflection", "Signage Detection"])
            interpretations = {
                "Micro-Texture": "Organic matter is identified by high-frequency spatial variation and lack of sharp edges.",
                "Geometric Reflection": "Model detects specular light on smooth surfaces (bottles/cans).",
                "Signage Detection": "CNN filters for standard high-contrast hazard labels."
            }
            st.info(interpretations[selected_feature])
        with feature_col2:
            # Multi-layer visualization simulation
            st.markdown("### Layer-wise attention (Activation Map Simulation)")
            st.image("https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png", 
                     caption="Top: Input | Bottom: Grad-CAM Overlay", use_column_width=True)

    except Exception as e:
        st.error(f"Failed to load analytics: {e}")

# 3. MLOps Tab
with tab_mlops:
    st.header("🚀 Automated Retraining Pipeline")
    st.markdown("""
    Trigger a model update by providing a new labeled dataset. 
    The system will automatically perform **Transfer Learning Fine-tuning** and version the new weights.
    """)
    
    with st.expander("Step 1: Upload Data"):
        new_zip = st.file_uploader("Upload .zip (must contain /train and /test folders)", type="zip")
        if new_zip:
            if st.button("Upload to Server"):
                with st.spinner("Uploading..."):
                    up_resp = requests.post(f"{API_URL}/upload-data", files={"file": (new_zip.name, new_zip.getvalue())})
                    if up_resp.status_code == 200:
                        st.success("Data staged successfully.")
                        st.session_state['data_ready'] = True
                    else:
                        st.error("Upload failed.")

    with st.expander("Step 2: Trigger Training"):
        if st.session_state.get('data_ready'):
            if st.button("Start Fine-Tuning Process"):
                retrain_resp = requests.post(f"{API_URL}/retrain")
                if retrain_resp.status_code == 200:
                    st.toast("Retraining Pipeline Started!")
                    st.info("The system is now fine-tuning in the background. Monitor progress in the sidebar.")
                else:
                    st.error("Could not start retraining.")
        else:
            st.write("Please upload data first.")
