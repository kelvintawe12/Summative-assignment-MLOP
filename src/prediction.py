import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import cv2
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

def load_trained_model(model_path="models/waste_model_v1.keras"):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def get_gradcam_overlay(model, img_array, intensity=0.5, res=224):
    """
    Generates a Grad-CAM heatmap overlay for the input image.
    """
    # 1. Find the last convolutional layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model): # It's the EfficientNet base
            for sub_layer in reversed(layer.layers):
                if "conv" in sub_layer.name.lower():
                    last_conv_layer_name = sub_layer.name
                    base_model = layer
                    break
            if last_conv_layer_name: break

    # 2. Create a model that maps input to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [base_model.inputs], [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    # 3. Compute Gradient
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 4. Multiply each channel by "how important it is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. Normalize and Process
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # 6. Overlay on original image
    img = img_array[0]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * intensity + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

CONFIDENCE_THRESHOLD = 0.5

def predict_image(model, image_path, class_names, generate_heatmap=True):
    """
    Performs inference with uncertainty detection.
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = tf.expand_dims(img_array, 0)

    # Predict
    predictions = model.predict(img_array_expanded, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    
    # Uncertainty Detection
    is_uncertain = confidence < CONFIDENCE_THRESHOLD
    
    result = {
        "class": class_names[class_idx] if not is_uncertain else "Uncertain/Unknown",
        "confidence": confidence,
        "is_uncertain": is_uncertain,
        "all_scores": {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    }

    if generate_heatmap:
        heatmap_img = get_gradcam_overlay(model, img_array_expanded)
        # Unique name to avoid collisions
        h_path = f"temp_heatmap_{int(time.time())}.jpg"
        cv2.imwrite(h_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
        result["heatmap_path"] = h_path

    return result

