import os
import requests

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
path = "credit_card_fraud_detection/data/creditcard.csv"

os.makedirs(os.path.dirname(path), exist_ok=True)
if not os.path.exists(path):
    print("Downloading dataset...")
    r = requests.get(url)
    with open(path, "wb") as f:
        f.write(r.content)
    print("Download complete.")
else:
    print("Dataset already exists.")
