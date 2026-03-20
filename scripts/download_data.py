import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def download_dataset():
    # Use PhenomSG's Waste Classification Dataset as recommended
    dataset_name = "phenomsg/waste-classification"
    target_dir = "data"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    logger.info(f"Downloading dataset {dataset_name} to {target_dir}...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", target_dir, "--unzip"], check=True)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.error("Please ensure you have the Kaggle API installed (`pip install kaggle`) and `kaggle.json` set up.")

if __name__ == "__main__":
    download_dataset()

