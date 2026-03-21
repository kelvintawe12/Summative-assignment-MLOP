import os
import tensorflow as tf
import shutil

def validate_tf_compatibility(directory):
    """
    Uses TensorFlow's own decoder to verify image compatibility.
    """
    invalid_files = []
    print(f"Deep scanning for TF compatibility: {directory}")
    
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
                
            file_path = os.path.join(root, file)
            try:
                # Read file as bytes
                img_bytes = tf.io.read_file(file_path)
                # Attempt to decode - this is where your error happens
                tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
            except Exception as e:
                print(f"Incompatible file found: {file_path}")
                invalid_files.append(file_path)

    return invalid_files

def run_cleanup():
    train_dir = "data/train"
    test_dir = "data/test"
    corrupt_dir = "data/corrupt_files"
    
    os.makedirs(corrupt_dir, exist_ok=True)
    
    all_invalid = []
    if os.path.exists(train_dir):
        all_invalid.extend(validate_tf_compatibility(train_dir))
    if os.path.exists(test_dir):
        all_invalid.extend(validate_tf_compatibility(test_dir))
        
    if all_invalid:
        print(f"\nFound {len(all_invalid)} files that TensorFlow cannot decode.")
        for f in all_invalid:
            # Move to corrupt folder to keep data clean
            filename = os.path.basename(f)
            # Prepend class name to filename to avoid collisions in corrupt_dir
            class_name = os.path.basename(os.path.dirname(f))
            dest = os.path.join(corrupt_dir, f"{class_name}_{filename}")
            shutil.move(f, dest)
            print(f"Isolated: {f} -> {dest}")
        print("\nCleanup complete. You can now restart your notebook kernel and train.")
    else:
        print("\nNo incompatible images found. Ensure your data/train and data/test paths are correct.")

if __name__ == "__main__":
    run_cleanup()
