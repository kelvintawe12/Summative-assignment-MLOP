import os
import shutil
import zipfile
import random

corrupt_dir = "data/corrupt_files"
output_dir = "data/retrain_data"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

categories = ["Hazardous", "Non-Recyclable", "Organic", "Recyclable"]

# Create category folders for train and test
def make_dirs():
    for cat in categories:
        os.makedirs(os.path.join(train_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cat), exist_ok=True)

make_dirs()

# Gather images by category
cat_files = {cat: [] for cat in categories}
for fname in os.listdir(corrupt_dir):
    for cat in categories:
        if fname.startswith(cat):
            cat_files[cat].append(fname)

# Split 80% train, 20% test and copy
for cat, files in cat_files.items():
    random.shuffle(files)
    split_idx = int(0.8 * len(files))
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    for fname in train_files:
        shutil.copy2(os.path.join(corrupt_dir, fname), os.path.join(train_dir, cat, fname))
    for fname in test_files:
        shutil.copy2(os.path.join(corrupt_dir, fname), os.path.join(test_dir, cat, fname))

# Zip the train and test folders
zip_path = "waste_retrain_data.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for folder in [train_dir, test_dir]:
        for root, _, files in os.walk(folder):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, output_dir)
                zipf.write(abs_path, rel_path)

print(f"Created {zip_path} for retraining!")
