# download.py
# Automatically downloads the House Prices dataset using the Kaggle API.
# The file will always be placed in <project_root>/data/ as train.csv

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# --------------------------------------------------------------------
# Always resolve paths relative to this file (not the terminal location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
# --------------------------------------------------------------------

def download_data():
    # Create data folder if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Download the train.csv file
    api.competition_download_file(
        competition="house-prices-advanced-regression-techniques",
        file_name="train.csv",
        path=DATA_DIR
    )

    # If Kaggle returned a ZIP, unzip it
    zip_path = os.path.join(DATA_DIR, "train.csv.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path)
        print("✅ train.csv downloaded and extracted to data/")
    else:
        print("✅ train.csv downloaded to data/ (no unzip needed)")

if __name__ == "__main__":
    download_data()
