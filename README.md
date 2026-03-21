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
- **Production Monitoring**: Real-time telemetry tracking of inference latency, confidence intervals, and system uptime.
- **Metadata Persistence**: Integrated SQLite database for tracking training history, model paths, and performance metrics.
- **Automated Quality Assurance**: Comprehensive suite of unit and integration tests using pytest.
- **Continuous Integration**: GitHub Actions workflow for automated testing and code validation on every push.

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
├── notebook/             # Research and Development (Advanced EDA, Training, Grad-CAM)
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


## Setup and Installation

### 1. Prerequisites
- Python 3.9 or higher (for local development)
- Docker and Docker-Compose (optional, for containerized deployment)
- Kaggle account and API credentials

### 2. Clone the Repository
```sh
git clone https://github.com/username/SmartWasteClassifier.git
cd SmartWasteClassifier
```

### 3. Create and Activate a Virtual Environment (Recommended)
```sh
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```sh
pip install -r requirements.txt
```

### 5. Configure Kaggle API Credentials
1. Go to your Kaggle account settings and create a new API token.
2. Download `kaggle.json` and place it in `~/.kaggle/kaggle.json`:
   ```sh
   mkdir -p ~/.kaggle
   mv /path/to/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 6. Download the Dataset
```sh
python scripts/download_data.py
# If you encounter issues, you can manually download and unzip:
kaggle datasets download -d phenomsg/waste-classification -p data
unzip -o data/waste-classification.zip -d data
```

### 7. Run Tests (Optional but Recommended)
```sh
PYTHONPATH=$(pwd) pytest tests/
```

### 8. Start the Application

#### Option A: Local Development
- Start the FastAPI backend:
  ```sh
  uvicorn api.main:app --reload
  ```
- In a new terminal, start the Streamlit web app:
  ```sh
  streamlit run app/app.py
  ```
- Access the UI at: http://localhost:8501
- Access the API docs at: http://localhost:8000/docs

#### Option B: Docker Compose (Full Stack)
```sh
docker-compose up --build
```
- **Administrative UI**: http://localhost:8501
- **Inference API**: http://localhost:8000

---

## Advanced MLOps Workflow

### 1. Robust Data Validation
The system implements structural validation for all uploaded datasets. ZIP files must contain specific `train/` and `test/` directory structures. Corrupt or incorrectly formatted data is automatically rejected before the retraining process begins.

### 2. Performance Benchmarking
To simulate high-traffic scenarios and verify system stability:
```bash
locust -f locustfile.py --host=http://localhost:8000
```
To scale the inference engine to 3 replicas:
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
- **Core Research**: See notebook/ directory for comprehensive development logs.

---
*This project was developed for the Introduction to Machine Learning Module Summative Assignment (2026).*
