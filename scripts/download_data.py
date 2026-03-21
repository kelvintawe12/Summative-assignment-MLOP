import os
import shutil
import random
import glob
import subprocess

def download_and_organize():
    """
    Downloads the PhenomSG dataset from Kaggle, unzips it, 
    and organizes it into train/test splits.
    """
    base_dir = "data"
    dataset_name = "phenomsg/waste-classification"
    zip_path = os.path.join(base_dir, "waste-classification.zip")
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 1. Download Dataset if not present
    # We check for the zip file or the final folders
    if not os.path.exists(os.path.join(base_dir, "train")):
        print(f"Downloading dataset {dataset_name}...")
        try:
            # Using subprocess to call kaggle CLI (requires kaggle.json)
            subprocess.run([
                "kaggle", "datasets", "download", "-d", dataset_name, 
                "-p", base_dir
            ], check=True)
            
            print("Download successful. Unzipping...")
            # 2. Unzip
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            print("Unzip complete.")
            
        except Exception as e:
            print(f"Error during download/unzip: {e}")
            print("Troubleshooting: Ensure 'pip install kaggle' is run and '~/.kaggle/kaggle.json' exists.")
            return

    # 3. Organize into Train/Test Splits
    classes = ["Hazardous", "Non-Recyclable", "Organic", "Recyclable"]
    train_base = os.path.join(base_dir, "train")
    test_base = os.path.join(base_dir, "test")

    print("Organizing files into train and test splits...")

    for cls in classes:
        # Search recursively for images in the unzipped folders
        # PhenomSG unzips into data/Hazardous/Hazardous/subfolders/*.jpg
        search_pattern = os.path.join(base_dir, cls, "**", "*.*")
        all_files = [f for f in glob.glob(search_pattern, recursive=True) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not all_files:
            continue

        os.makedirs(os.path.join(train_base, cls), exist_ok=True)
        os.makedirs(os.path.join(test_base, cls), exist_ok=True)

        random.seed(42)
        random.shuffle(all_files)
        split_idx = int(0.8 * len(all_files))
        
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]

        # Move/Copy to final structure
        for f in train_files:
            rel_path = os.path.relpath(f, os.path.join(base_dir, cls))
            unique_name = rel_path.replace(os.sep, "_")
            shutil.move(f, os.path.join(train_base, cls, unique_name))
            
        for f in test_files:
            rel_path = os.path.relpath(f, os.path.join(base_dir, cls))
            unique_name = rel_path.replace(os.sep, "_")
            shutil.move(f, os.path.join(test_base, cls, unique_name))

        # Cleanup the now-empty original class folder
        original_cls_path = os.path.join(base_dir, cls)
        if os.path.exists(original_cls_path):
            shutil.rmtree(original_cls_path)

    # Final cleanup of zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("\nProcess Complete:")
    print(f"Data Location: {os.path.abspath(base_dir)}")
    print("Structure: 'data/train' and 'data/test' created successfully.")

if __name__ == "__main__":
    download_and_organize()
