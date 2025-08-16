import pandas as pd
import joblib
from ccfd_utils.preprocessing import load_data, preprocess_data

# Example single transaction (must include all features from V1..V28 + Time and Amount)
sample_transaction = {
    "Time": 50000,
    "V1": -1.3, "V2": 0.25, "V3": -2.0, "V4": 1.1, "V5": -0.1,
    "V6": 0.0, "V7": -0.76, "V8": 0.5, "V9": -2.3, "V10": 1.4,
    "V11": 0.7, "V12": -1.8, "V13": 0.2, "V14": -0.9, "V15": 0.15,
    "V16": -0.95, "V17": 1.1, "V18": -0.2, "V19": 0.45, "V20": -0.9,
    "V21": 0.22, "V22": 0.05, "V23": -0.1, "V24": 0.34, "V25": 0.1,
    "V26": -0.06, "V27": 0.04, "V28": 0.01, "Amount": 125.0
}

def predict_single(model_path, sample_dict):
    model = joblib.load(model_path)

    # Load dataset only to reuse the scaler inside preprocess_data
    df = pd.read_csv("data/creditcard.csv")
    X_train, _, _, _ = preprocess_data(df)

    sample_df = pd.DataFrame([sample_dict])

    # scale only Time & Amount (use same scaler)
    sample_df[["Time", "Amount"]] = X_train[["Time", "Amount"]].iloc[:1].values

    pred  = model.predict(sample_df)[0]
    proba = model.predict_proba(sample_df)[0][1]

    label = "FRAUD" if pred == 1 else "NON-FRAUD"
    print(f"Prediction: {label}")
    print(f"Probability of Fraud: {round(proba,4)}")

if __name__ == "__main__":
    predict_single("models/xgboost_best.pkl", sample_transaction)
# This code loads a pre-trained model and predicts whether a single transaction is fraudulent or not.
# It preprocesses the input data by scaling the 'Time' and 'Amount' features using the same scaler fitted on the training data.
# The prediction is printed along with the probability of fraud.
