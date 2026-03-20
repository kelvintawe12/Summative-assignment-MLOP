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
import base64

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Waste Classifier Pro API")

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
    "prediction_history": [],
    "last_upload_path": None
}

@app.on_event("startup")
async def startup_event():
    state["model"] = load_trained_model(MODEL_PATH)
    logger.info("Model loaded successfully")

@app.get("/health")
async def health_check():
    model_status = "loaded" if state["model"] is not None else "not_loaded"
    uptime = time.time() - START_TIME
    return {
        "status": "up" if model_status == "loaded" else "degraded",
        "model_version": state["model_path"],
        "uptime": f"{uptime:.0f}s",
        "predictions_processed": len(state["prediction_history"])
    }

@app.get("/stats")
async def get_stats():
    # Placeholder for dataset stats - in production would query DB
    return {
        "dataset": {
            "classes": CLASS_NAMES,
            "train_count": 24000,
            "test_count": 6000
        },
        "history": [] # Would load from CSV or DB
    }

@app.get("/history")
def get_prediction_history():
    return state["prediction_history"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not state["model"]:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    start_time = time.time()
    temp_path = f"temp_{int(time.time())}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = predict_image(state["model"], temp_path, CLASS_NAMES)
        latency = time.time() - start_time
        
        # Log to history
        state["prediction_history"].append({
            "timestamp": time.strftime("%H:%M:%S"),
            "class": result["class"],
            "latency": latency,
            "confidence": result["confidence"]
        })
        if len(state["prediction_history"]) > 50: state["prediction_history"].pop(0)

        with open(result["heatmap_path"], "rb") as h_file:
            encoded_heatmap = base64.b64encode(h_file.read()).decode('utf-8')

        return {
            "class_name": result["class"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
            "heatmap_base64": encoded_heatmap,
            "latency": latency
        }
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        if os.path.exists("temp_heatmap.jpg"): os.remove("temp_heatmap.jpg")

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip file.")
        
    upload_dir = "data/new_uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        
    file_path = os.path.join(upload_dir, f"retrain_{int(time.time())}.zip")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Store path in state for retraining
    state["last_upload_path"] = file_path
    return {"message": f"Successfully uploaded {file.filename} for retraining.", "path": file_path}

@app.post("/retrain", response_model=RetrainResponse)
async def trigger_retrain(background_tasks: BackgroundTasks):
    if state["is_retraining"]:
        return {"status": "already_running", "message": "A retraining task is already in progress."}
    
    if "last_upload_path" not in state or not os.path.exists(state["last_upload_path"]):
        raise HTTPException(status_code=400, detail="No new data found. Please upload a ZIP file first.")

    def run_retraining_task(zip_path):
        state["is_retraining"] = True
        state["last_retrain_status"] = "running"
        try:
            new_path, _ = retrain_existing_model(state["model_path"], zip_path, DATA_DIR)
            state["model_path"] = new_path
            state["model"] = load_trained_model(new_path)
            state["last_retrain_status"] = "success"
            logger.info(f"Background retraining successful. New model: {new_path}")
        except Exception as e:
            state["last_retrain_status"] = f"failed: {str(e)}"
            logger.error(f"Background retraining failed: {e}")
        finally:
            state["is_retraining"] = False

    background_tasks.add_task(run_retraining_task, state["last_upload_path"])
    return {"status": "started", "message": "Retraining task triggered in background."}

@app.get("/retrain/status")
def get_retrain_status():
    return {
        "is_retraining": state["is_retraining"],
        "last_status": state["last_retrain_status"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
