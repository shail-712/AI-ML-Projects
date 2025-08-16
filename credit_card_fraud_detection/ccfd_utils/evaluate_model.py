import pandas as pd
import joblib
from ccfd_utils.preprocessing import load_data, preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """Evaluate the given model on the test data and print metrics."""
    predictions = model.predict(X_test)
    proba       = model.predict_proba(X_test)[:, 1]   # for AUC
    
    print(f"Model: {model.__class__.__name__}")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print("-" * 60)

def load_and_evaluate_models(model_paths, X_test, y_test):
    """Load models from the specified paths and evaluate them on the test data."""
    for model_path in model_paths:
        model = joblib.load(model_path)
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data("../data/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Define model paths
    model_paths = ["../models/random_forest.pkl", "../models/xgboost_best.pkl"]

    # Load and evaluate models
    load_and_evaluate_models(model_paths, X_test, y_test)
