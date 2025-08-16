import pandas as pd
import joblib
from preprocessing import load_data, preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Load the final model
    model = joblib.load("models/xgboost_best.pkl")

    # Evaluate
    predictions = model.predict(X_test)
    probability  = model.predict_proba(X_test)[:, 1]

    print(f"Model: {model.__class__.__name__}")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probability))
