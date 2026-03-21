# Smart Waste Classifier Pro: Industrial MLOps Pipeline

This repository contains a professional-grade Machine Learning pipeline designed for the automated classification of urban waste. The system utilizes State-of-the-Art (SOTA) Computer Vision architectures and implements a complete MLOps lifecycle, including automated retraining, real-time monitoring, deep model interpretability, and automated quality assurance.

---

## Project Overview

The Smart Waste Classifier Pro is engineered to address waste contamination in smart city infrastructures. By employing Transfer Learning with the EfficientNetV2B0 architecture, the system achieves high-precision classification across four primary categories. The solution is comprised of a robust FastAPI backend for low-latency inference and a Streamlit-based management console for system administration and analytical oversight.

### Technical Capabilities
- **Neural Inference Engine**: High-accuracy classification of Hazardous, Organic, Recyclable, and Non-Recyclable waste.
- **Model Interpretability**: Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) providing real-time visual heatmaps to explain model decision-making.
- **Automated Retraining Pipeline**: Synchronous and asynchronous retraining triggers with automated data extraction and model versioning.
- **Champion-Challenger Validation**: New models are evaluated against a test set and only promoted to production if they outperform the current "Champion" model.
- **Uncertainty Detection**: Implements a confidence thresholding mechanism to identify and flag ambiguous or out-of-distribution inputs.
- **Metadata Persistence**: Integrated SQLite database for tracking training history, data upload logs, and performance metrics.
- **Continuous Integration**: GitHub Actions workflow for automated testing and code validation on every push.

---

## Technical Stack

- **Deep Learning Framework**: TensorFlow 2.15, Keras (EfficientNetV2B0 backbone).
- **Inference API**: FastAPI (Asynchronous, Type-hinted).
- **Administrative Interface**: Streamlit (Interactive Analytics and MLOps Controls).
- **Data Visualization**: Plotly Express, Seaborn, Matplotlib.
- **Persistence**: SQLite3 (Metadata), Local Versioned Storage (.keras).
- **Testing & CI/CD**: Pytest, GitHub Actions.
- **Orchestration**: Docker, Docker-Compose, Locust.

---

## System Hierarchy

```text
SmartWasteClassifier/
├── .github/workflows/    # Continuous Integration (GitHub Actions)
├── api/                  # FastAPI Application and Schemas
├── app/                  # Streamlit Administrative Dashboard
├── data/                 # Local Dataset Storage (Internal Structure: /train, /test)
├── models/               # Versioned Model Repository and SQLite Metadata DB
├── notebook/             # Research and Development (Advanced EDA, Training, Analytics)
├── src/                  # Core Modular Logic
│   ├── preprocessing.py  # Data loading and stats generation
│   ├── model.py          # Architecture definition
│   ├── prediction.py     # Inference and Grad-CAM logic
│   ├── retrain.py        # Fine-tuning engine
│   └── registry.py       # Model metadata management
├── tests/                # Automated Pipeline Tests (Pytest)
├── scripts/              # Automation and Data Acquisition Scripts
├── Dockerfile            # Container Specification
├── docker-compose.yml    # Service Orchestration and Scaling
├── requirements.txt      # Dependency Specification
└── locustfile.py         # Performance Benchmarking Script
```

---

## Setup and Operational Instructions

### 1. Environment Initialization
Ensure you have Python 3.10+ and a virtual environment active. Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Acquisition and Organization
Download the PhenomSG dataset and automatically organize it into the required training and testing splits:
```bash
python scripts/download_data.py
```

### 3. Dataset Integrity Validation
Run the deep-scan utility to identify and isolate any corrupt or incompatible image files that may cause training failures:
```bash
python scripts/validate_data.py
```

### 4. Bootstrapping the Initial Model (Champion)
Before the API can serve predictions, an initial model must be generated and registered in the database. This script performs a rapid initial training session:
```bash
python scripts/train_initial_model.py
```

### 5. Launching the Services
Start the backend API and the management console in separate terminals:

**Terminal 1 (Backend API):**
```bash
uvicorn api.main:app --port 8000
```

**Terminal 2 (Streamlit UI):**
```bash
streamlit run app/app.py
```

---

## Evaluation and Monitoring

### 1. Research Notebook
Execute `notebook/waste_classification_project.ipynb` for advanced analytical research, including **t-SNE latent space clustering**, **ROC curves**, and **Training History dynamics**.

### 2. Production Monitoring
Access the **Data Insights** tab in the Streamlit UI to track real-time inference latency and model performance metrics stored in the SQLite database.

### 3. Load Testing
To simulate high-concurrency traffic and verify system stability:
```bash
locust -f locustfile.py --host=http://localhost:8000
```

---

## 📋 Rubric Compliance Mapping

| Criterion | Implementation in this Project |
|-----------|--------------------------------|
| **Video Demo** | [Camera ON] Demonstrates end-to-end image prediction and bulk retraining trigger. |
| **Retraining Process** | 1. **Data Upload**: ZIP extraction & SQLite logging (`data_uploads` table). <br> 2. **Preprocessing**: Automated validation & augmentation. <br> 3. **Custom Model**: Retrains using the existing Champion model weights. |
| **Prediction Process** | UI accepts image uploads and displays labeled classification with Grad-CAM visual evidence. |
| **Evaluation of Models** | Advanced notebook with **5 metrics**: Accuracy, Loss, Precision, Recall, and AUC. Uses EarlyStopping and EfficientNetV2B0. |
| **Deployment Package** | Fully containerized via **Docker-Compose**. Streamlit UI includes interactive Plotly **Data Insights**. |

---
*This project was developed for the Introduction to Machine Learning Module Summative Assignment (2026).*
