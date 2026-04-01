import tensorflow as tf
from src.preprocessing import get_data_loaders
from src.model import train_model
import os
import shutil
import zipfile
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_zip(zip_path, extract_to):
    """
    Safely extracts uploaded ZIP containing new data.
    """
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info("Extraction complete.")

def retrain_existing_model(model_path, uploaded_zip_path, base_data_dir, epochs=5):
    """
    Robust retraining: 
    1. Extracts new data
    2. Loads model
    3. Unfreezes specific layers for fine-tuning
    4. Trains and saves versioned model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_path = f"models/waste_model_{timestamp}.keras"
    temp_extract_dir = f"data/temp_retrain_{timestamp}"
    use_temp_data = uploaded_zip_path is not None
    if use_temp_data and not os.path.exists(temp_extract_dir):
        os.makedirs(temp_extract_dir)

    try:
        if use_temp_data:
            # 1. Extract and Validate
            extract_zip(uploaded_zip_path, temp_extract_dir)
            # 2. Setup loaders for new data (merging logic can be added here)
            train_ds, test_ds, class_names = get_data_loaders(
                train_dir=os.path.join(temp_extract_dir, 'train'),
                test_dir=os.path.join(temp_extract_dir, 'test')
            )
        else:
            logger.info("No new dataset provided. Using current data directory for retraining.")
            train_ds, test_ds, class_names = get_data_loaders(
                train_dir=os.path.join(base_data_dir, 'train'),
                test_dir=os.path.join(base_data_dir, 'test')
            )

        # 3. Load & Modify Model
        logger.info(f"Fine-tuning model: {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Identify the base model (EfficientNet)
        base_model_layer = None
        for layer in model.layers:
            if "efficientnetv2-b0" in layer.name.lower():
                base_model_layer = layer
                break
        
        if base_model_layer:
            logger.info("Unfreezing last 30 layers for fine-tuning...")
            base_model_layer.trainable = True
            # Freeze all layers except last 30
            for layer in base_model_layer.layers[:-30]:
                layer.trainable = False
            for layer in base_model_layer.layers[-30:]:
                layer.trainable = True

        # 4. Recompile with lower LR
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 5. Train
        history = train_model(
            model, 
            train_ds, 
            test_ds, 
            epochs=epochs, 
            model_save_path=new_model_path
        )
        
        # 6. Final Evaluation for Champion Validation
        logger.info("Evaluating Challenger model for Champion validation...")
        eval_results = model.evaluate(test_ds, return_dict=True)
        
        logger.info(f"Retraining successful. New model: {new_model_path}")
        return new_model_path, eval_results

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise e
    finally:
        # Cleanup temp extraction
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
