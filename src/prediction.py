import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.cm as cm
import time
import os

def load_trained_model(model_path="models/waste_model_v1.keras"):
    try:
        if not os.path.exists(model_path):
            return None
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_gradcam_overlay(model, img_array, intensity=0.5):
    """
    Robust Grad-CAM for nested EfficientNet models.
    """
    try:
        # Find the base model layer
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) or "efficientnet" in layer.name.lower():
                base_model = layer
                break
        
        if not base_model:
            return None

        # Find last conv layer in the base model
        last_conv_layer_name = None
        for layer in reversed(base_model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        
        if not last_conv_layer_name:
            return None

        # Create a mapping from input to the conv layer and the final output
        # We use model.input to ensure we cover the entire graph
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [base_model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        img = img_array[0]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap * intensity + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        return superimposed_img
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return None

CONFIDENCE_THRESHOLD = 0.5

def predict_image(model, image_path, class_names, generate_heatmap=True):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array_expanded, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    
    is_uncertain = confidence < CONFIDENCE_THRESHOLD
    
    result = {
        "class": class_names[class_idx] if not is_uncertain else "Uncertain/Unknown",
        "confidence": confidence,
        "is_uncertain": is_uncertain,
        "all_scores": {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    }

    if generate_heatmap:
        heatmap_img = get_gradcam_overlay(model, img_array_expanded)
        if heatmap_img is not None:
            h_path = f"temp_heatmap_{int(time.time())}.jpg"
            cv2.imwrite(h_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
            result["heatmap_path"] = h_path

    return result
