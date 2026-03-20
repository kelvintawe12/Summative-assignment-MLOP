import tensorflow as tf
import os
import pandas as pd
from PIL import Image

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_dataset_stats(base_dir):
    """
    Analyzes the dataset directory to provide class counts and metadata.
    """
    stats = []
    if not os.path.exists(base_dir):
        return pd.DataFrame()

    for split in ['train', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                stats.append({
                    "split": split,
                    "class": class_name,
                    "count": count
                })
    return pd.DataFrame(stats)

def get_data_loaders(train_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    # Training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    # Testing dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Class names for reference
    class_names = train_ds.class_names
    
    # Configure for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds, class_names

def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])
