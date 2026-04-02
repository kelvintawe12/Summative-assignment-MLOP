import os
import sys

# Add project root to path so src/ can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from src.preprocessing import get_data_loaders
from src.model import build_model, train_model
import sqlite3
import time
import numpy as np

def bootstrap_initial_model():
    """
    Trains the very first model version so the API and UI are functional.
    """
    train_dir = "data/train"
    test_dir = "data/test"
    model_dir = "models"
    model_path = os.path.join(model_dir, "waste_model_v1.keras")
    db_path = os.path.join(model_dir, "model_metadata.db")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("Error: Dataset folders not found. Please run 'python scripts/download_data.py' first.")
        return

    os.makedirs(model_dir, exist_ok=True)

    print("Loading datasets...")
    train_ds, test_ds, class_names = get_data_loaders(train_dir, test_dir)

    print(f"Building model for {len(class_names)} classes: {class_names}")
    model = build_model(num_classes=len(class_names))

    print("Starting initial training (3 epochs for rapid setup)...")
    history = train_model(model, train_ds, test_ds, epochs=3, model_save_path=model_path)

    print("Recording model metrics in database...")
    # Find the index of the best epoch to ensure DB metrics match saved weights
    best_idx = np.argmax(history.history['val_accuracy'])
    accuracy = history.history['val_accuracy'][best_idx]
    loss = history.history['val_loss'][best_idx]
    
    # Initialize database and record metrics
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Ensure schema exists before insertion
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                file_path TEXT,
                file_size_kb REAL
            )
        ''')

        cursor.execute('''
            INSERT INTO training_history (timestamp, model_path, accuracy, loss, status, is_champion)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (time.ctime(), model_path, float(accuracy), float(loss), 'success', 1))
        conn.commit()

    print(f"\nSuccess! Initial model saved to {model_path}")
    print("Next Steps:")
    print("1. Start API: uvicorn api.main:app --port 8000")
    print("2. Start UI: streamlit run app/app.py")

if __name__ == "__main__":
    bootstrap_initial_model()
