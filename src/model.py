import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_model(num_classes, img_size=(224, 224, 3)):
    """
    Builds a robust transfer learning model using EfficientNetV2B0.
    Includes BatchNormalization and Dropout for better generalization.
    """
    logger.info(f"Building model for {num_classes} classes with input shape {img_size}")
    
    # Base model
    base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=img_size)
    base_model.trainable = False # Freeze by default

    # Top layers
    inputs = Input(shape=img_size)
    # EfficientNetV2 expects [0, 255] range, so we don't need manual scaling if using the internal preprocessing
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    
    # Compile with multiple metrics for better evaluation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def train_model(model, train_ds, test_ds, epochs=20, model_save_path="models/waste_model_v1.keras"):
    """
    Premium training loop with EarlyStopping, ReduceLROnPlateau, and CSVLogging.
    """
    if not os.path.exists('models'):
        os.makedirs('models')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, 
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            model_save_path.replace('.keras', '_history.csv'), 
            append=True
        )
    ]

    logger.info(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history
