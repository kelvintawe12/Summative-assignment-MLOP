import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress noisy TensorFlow/CUDA warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import shutil
import logging
import threading # Added threading import
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

# Global state for log buffer
_LOG_BUFFER = []
_LOG_LOCK = threading.Lock()

class BufferLogHandler(logging.Handler):
    def emit(self, record):
        with _LOG_LOCK:
            formatted_message = self.format(record) # This populates record.asctime
            _LOG_BUFFER.append({
                "timestamp": record.asctime,
                "level": record.levelname,
                "message": record.message, # Raw message
                "full_formatted": formatted_message # Fully formatted string
            })
            if len(_LOG_BUFFER) > 30:
                _LOG_BUFFER.pop(0)

buffer_handler = BufferLogHandler()
buffer_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(buffer_handler)

app = FastAPI(title="Smart Waste Classifier Pro API")

# Database Initialization
DB_PATH = os.getenv("DB_PATH", "models/model_metadata.db")

def get_db_connection():
    """Establishes a connection to the metadata database."""
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the model registry and data logs in SQLite."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Model Registry Table: Tracks all training iterations and active Champion
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_path TEXT,
                accuracy REAL,
                loss REAL,
                status TEXT,
                is_champion INTEGER DEFAULT 0
            )
        ''')
        # Data Management Table: Tracks bulk data ingestion events
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

init_db()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint: API description, endpoint listing, and recent logs
@app.get("/")
def root():
    with _LOG_LOCK:
        logs = list(_LOG_BUFFER)
    return {
        "title": "Smart Waste Classifier Pro API",
        "description": "A FastAPI backend for waste classification, retraining, and analytics.",
        "endpoints": {
            "/health": "Get API health, uptime, and model status.",
            "/stats": "Get dataset and training history statistics.",
            "/history": "Get recent prediction history.",
            "/predict": "POST: Predict waste class from an uploaded image.",
            "/upload-data": "POST: Upload new training data as a .zip file.",
            "/retrain": "POST: Trigger model retraining (background).",
            "/retrain/status": "Get retraining status and model registry info."
        },
        "recent_logs": logs
    }


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
    # Optimization: Prevent TensorFlow from pre-allocating all memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"GPU config error: {e}")

    # Attempt to locate the Champion model in the registry
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT model_path FROM training_history WHERE is_champion=1 AND status='success' ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                state["model_path"] = row[0]
                logger.info(f"Registry Lookup: Active Champion found at {state['model_path']}")
        except Exception as e:
            logger.warning(f"Registry Lookup: Error querying model metadata: {e}")

    if os.path.exists(state["model_path"]):
        logger.info(f"Loading model from {state['model_path']}...")
        try:
            state["model"] = load_trained_model(state["model_path"])
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning(f"Model file not found at {state['model_path']}. API starting in degraded mode.")

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
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO data_uploads (timestamp, filename, file_path, file_size_kb)
            VALUES (?, ?, ?, ?)
        ''', (time.ctime(), file.filename, file_path, file_size))
        conn.commit()

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
            
            # Update Model Registry: Set new model as Champion and demote others
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE training_history SET is_champion = 0")
                cursor.execute('''
                    INSERT INTO training_history (timestamp, model_path, accuracy, loss, status, is_champion)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (time.ctime(), new_path, eval_metrics.get('accuracy'), eval_metrics.get('loss'), 'success', 1))
                conn.commit()

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
    with get_db_connection() as conn:
        history = pd.read_sql_query("SELECT * FROM training_history ORDER BY id DESC", conn).to_dict(orient="records")
        uploads = pd.read_sql_query("SELECT * FROM data_uploads ORDER BY id DESC", conn).to_dict(orient="records")
    
    champion = next((h for h in history if h.get('is_champion') == 1), history[0] if history else None)

    return {
        "is_retraining": state["is_retraining"], 
        "last_status": state["last_retrain_status"],
        "registry": {"history": history, "champion": champion, "uploads": uploads}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
