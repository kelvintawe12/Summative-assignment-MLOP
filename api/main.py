from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import shutil
import logging
import tensorflow as tf
from src.prediction import load_trained_model, predict_image
from src.retrain import retrain_existing_model
from api.schemas import PredictionResponse, HealthResponse, RetrainResponse
import numpy as np
from src.preprocessing import get_dataset_stats
import pandas as pd
from fastapi.responses import FileResponse
import base64
import sqlite3
import zipfile

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Waste Classifier Pro API")

# Database Initialization
DB_PATH = "models/model_metadata.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Table for Retraining History
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_path TEXT,
            accuracy REAL,
            loss REAL,
            status TEXT
        )
    ''')
    # Table for Data Upload Tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            file_path TEXT,
            file_size_kb REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/waste_model_v1.keras")
DATA_DIR = "data"
CLASS_NAMES = ["Hazardous", "Non-Recyclable", "Organic", "Recyclable"]
START_TIME = time.time()

# Persistent Global State
state = {
    "model": None,
    "model_path": MODEL_PATH,
    "is_retraining": False,
    "last_retrain_status": "none",
    "prediction_history": [] 
}

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info(f"Loading model from {state['model_path']}...")
    state["model"] = load_trained_model(state["model_path"])
    if state["model"] is None:
        logger.warning("No pre-trained model found. Initial training required.")

@app.get("/health", response_model=HealthResponse)
def health():
    uptime_seconds = time.time() - START_TIME
    uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"
    return {
        "status": "up" if state["model"] else "degraded",
        "uptime": uptime_str,
        "model_version": os.path.basename(state["model_path"])
    }

@app.get("/stats")
def get_stats():
    """Provides dataset and training history stats."""
    dataset_stats = get_dataset_stats(DATA_DIR)
    history_path = state["model_path"].replace('.keras', '_history.csv')
    history_data = []
    if os.path.exists(history_path):
        try:
            df_hist = pd.read_csv(history_path)
            history_data = df_hist.to_dict(orient="records")
        except:
            pass
    return {
        "dataset": dataset_stats.to_dict(orient="records") if not dataset_stats.empty else [],
        "history": history_data
    }

@app.get("/history")
def get_prediction_history():
    return state["prediction_history"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not state["model"]:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
        
    start_time_exec = time.time()
    temp_path = f"temp_{int(time.time())}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = {}
    try:
        result = predict_image(state["model"], temp_path, CLASS_NAMES)
        latency = time.time() - start_time_exec
        
        # Log to history
        state["prediction_history"].append({
            "timestamp": time.strftime("%H:%M:%S"),
            "class": result["class"],
            "latency": latency,
            "confidence": result["confidence"],
            "is_uncertain": result.get("is_uncertain", False)
        })
        if len(state["prediction_history"]) > 50: state["prediction_history"].pop(0)

        encoded_heatmap = None
        if "heatmap_path" in result and result["heatmap_path"] and os.path.exists(result["heatmap_path"]):
            with open(result["heatmap_path"], "rb") as h_file:
                encoded_heatmap = base64.b64encode(h_file.read()).decode('utf-8')

        return {
            "class_name": result["class"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
            "heatmap_base64": encoded_heatmap,
            "latency": latency,
            "is_uncertain": result.get("is_uncertain", False)
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        if "heatmap_path" in result and result.get("heatmap_path") and os.path.exists(result["heatmap_path"]):
            os.remove(result["heatmap_path"])

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip file.")
    upload_dir = "data/new_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"retrain_{int(time.time())}.zip")
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Database Persistence for Upload
    file_size = os.path.getsize(file_path) / 1024.0 # KB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO data_uploads (timestamp, filename, file_path, file_size_kb)
        VALUES (?, ?, ?, ?)
    ''', (time.ctime(), file.filename, file_path, file_size))
    conn.commit()
    conn.close()

    # Structural Validation
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            if not any('train/' in f for f in files) or not any('test/' in f for f in files):
                os.remove(file_path)
                raise HTTPException(status_code=400, detail="ZIP must contain 'train/' and 'test/' folders.")
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid ZIP: {str(e)}")

    state["last_upload_path"] = file_path
    return {"message": f"Successfully uploaded and recorded {file.filename}.", "path": file_path}

@app.post("/retrain", response_model=RetrainResponse)
async def trigger_retrain(background_tasks: BackgroundTasks):
    if state["is_retraining"]:
        return {"status": "already_running", "message": "A retraining task is already in progress."}

    # Use last_upload_path if available, else None (signals to use current dataset)
    zip_path = state.get("last_upload_path")
    if zip_path is not None and not os.path.exists(zip_path):
        zip_path = None

    def run_retraining_task(zip_path):
        state["is_retraining"] = True
        state["last_retrain_status"] = "running"
        try:
            new_path, eval_metrics = retrain_existing_model(state["model_path"], zip_path, DATA_DIR)
            # Database Persistence for Retraining
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_history (timestamp, model_path, accuracy, loss, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (time.ctime(), new_path, eval_metrics.get('accuracy'), eval_metrics.get('loss'), 'success'))
            conn.commit()
            conn.close()

            state["model_path"] = new_path
            state["model"] = load_trained_model(new_path)
            state["last_retrain_status"] = "success"
        except Exception as e:
            state["last_retrain_status"] = f"failed: {str(e)}"
        finally:
            state["is_retraining"] = False

    background_tasks.add_task(run_retraining_task, zip_path)
    return {"status": "started", "message": "Retraining task triggered in background."}

@app.get("/retrain/status")
def get_retrain_status():
    conn = sqlite3.connect(DB_PATH)
    history = pd.read_sql_query("SELECT * FROM training_history ORDER BY id DESC", conn).to_dict(orient="records")
    uploads = pd.read_sql_query("SELECT * FROM data_uploads ORDER BY id DESC", conn).to_dict(orient="records")
    conn.close()
    return {
        "is_retraining": state["is_retraining"], 
        "last_status": state["last_retrain_status"],
        "registry": {"history": history, "champion": history[0] if history else None, "uploads": uploads}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
