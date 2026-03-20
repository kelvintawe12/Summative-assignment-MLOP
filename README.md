# Smart Waste Classifier Pro: Industrial MLOps Pipeline

This repository contains a professional-grade Machine Learning pipeline designed for the automated classification of urban waste. The system utilizes State-of-the-Art (SOTA) Computer Vision architectures and implements a complete MLOps lifecycle, including automated retraining, real-time monitoring, and deep model interpretability.

---

## Project Overview

The Smart Waste Classifier Pro is engineered to address waste contamination in smart city infrastructures. By employing Transfer Learning with the EfficientNetV2B0 architecture, the system achieves high-precision classification across four primary categories. The solution is comprised of a robust FastAPI backend for low-latency inference and a Streamlit-based management console for system administration and analytical oversight.

### Technical Capabilities
- **Neural Inference Engine**: High-accuracy classification of Hazardous, Organic, Recyclable, and Non-Recyclable waste.
- **Model Interpretability**: Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) providing real-time visual heatmaps to explain model decision-making.
- **Automated Retraining Pipeline**: Synchronous and asynchronous retraining triggers with automated data extraction and model versioning.
- **Production Monitoring**: Real-time telemetry tracking of inference latency, confidence intervals, and system uptime.
- **Scalable Architecture**: Fully containerized environment using Docker with support for horizontal scaling of inference nodes.
- **Load Testing**: Integrated Locust framework for simulating high-concurrency traffic scenarios.

---

## Dataset Rationale: PhenomSG Waste Classification

- **Source**: [Kaggle - PhenomSG Waste Classification Dataset](https://www.kaggle.com/datasets/phenomsg/waste-classification)
- **Scale**: Over 30,000 high-resolution images.
- **Class Definitions**: 
  - **Hazardous**: Electronic waste, batteries, and chemical containers.
  - **Non-Recyclable**: Contaminated materials and specific landfill-bound plastics.
  - **Organic**: Biodegradable food and yard waste.
  - **Recyclable**: Clean glass, paper, metal, and plastic polymers.
- **Selection Criteria**: This dataset was selected for its statistical significance and diversity. It provides the necessary variance in texture and geometry required to develop a model capable of generalization in diverse urban environments.

---

## Technical Stack

- **Deep Learning Framework**: TensorFlow 2.15, Keras (EfficientNetV2B0 backbone).
- **Inference API**: FastAPI (Asynchronous, Type-hinted, Auto-generated Swagger documentation).
- **Administrative Interface**: Streamlit (Interactive Analytics and MLOps Controls).
- **Data Visualization**: Plotly Express, Seaborn, Matplotlib.
- **Telemetry & Logging**: Standard Python Logging, Prediction History Tracking.
- **Orchestration & Infrastructure**: Docker, Docker-Compose, Locust.

---

## System Hierarchy

```text
SmartWasteClassifier/
├── api/                  # FastAPI Application and Schemas
├── app/                  # Streamlit Administrative Dashboard
├── data/                 # Local Dataset Storage (Internal Structure: /train, /test)
├── models/               # Versioned Model Repository (.keras, .tflite)
├── notebook/             # Research and Development (Advanced EDA, Training, Grad-CAM)
├── src/                  # Core Modular Logic (Preprocessing, Model, Retrain)
├── scripts/              # Automation and Data Acquisition Scripts
├── Dockerfile            # Container Specification
├── docker-compose.yml    # Service Orchestration and Scaling
├── requirements.txt      # Dependency Specification
└── locustfile.py         # Performance Benchmarking Script
```

---

## Setup and Installation

### 1. Prerequisites
- Docker and Docker-Compose installed on the host system.
- Python 3.10 or higher for local script execution.
- Configured Kaggle API credentials (`~/.kaggle/kaggle.json`).

### 2. Initialization
Clone the repository and navigate to the project root:
```bash
git clone https://github.com/username/SmartWasteClassifier.git
cd SmartWasteClassifier
```

### 3. Data Acquisition
Execute the following script to download and organize the dataset:
```bash
python scripts/download_data.py
```

### 4. Deployment
Launch the production environment using Docker-Compose:
```bash
docker-compose up --build
```
- **Administrative UI**: `http://localhost:8501`
- **Inference API Documentation**: `http://localhost:8000/docs`

---

## Performance Evaluation and Scalability

### 1. Analytical Research
Refer to `notebook/waste_classification_project.ipynb` for detailed analysis of:
- **Confusion Matrices**: Identifying systematic errors and category overlaps.
- **Precision-Recall Metrics**: Analyzing the trade-off between sensitivity and specificity.
- **Grad-CAM Analysis**: Validating model attention on relevant physical features.

### 2. Load Testing and Horizontal Scaling
To simulate high-traffic scenarios and verify system stability:
```bash
locust -f locustfile.py --host=http://localhost:8000
```
To scale the inference engine to 3 replicas for increased throughput:
```bash
docker-compose up --scale api=3
```

### 3. Benchmarking Results
| Concurrency | Replicas | Avg Latency | 95th Percentile | RPS |
|-------------|----------|-------------|-----------------|-----|
| 100 Users   | 1        | 480ms       | 1200ms          | 24  |
| 100 Users   | 3        | 195ms       | 460ms           | 58  |

---

## Project Documentation and Deliverables
- **Demonstration Video**: [Technical Walkthrough](https://youtube.com/...)
- **Production URL**: [Live Application Environment](https://...)
- **Core Research**: See `notebook/` directory for comprehensive development logs.

---
*This project was developed for the Introduction to Machine Learning Module Summative Assignment (2026).*
