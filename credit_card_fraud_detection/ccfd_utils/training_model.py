import pandas as pd 
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from ccfd_utils.preprocessing import load_data, preprocess_data

def train_and_save():
    # Load and preprocess the data
    df = load_data("../data/creditcard.csv")
    # df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    # Train a Random Forest model
    rf_model = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print("Random Forest:\n", classification_report(y_test, rf_pred))
    print(confusion_matrix(y_test, rf_pred))
    # Train an XGBoost model

    xgb_model = XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]), use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    print("XGBoost:\n", classification_report(y_test, xgb_pred))
    print(confusion_matrix(y_test, xgb_pred))

    # Save the models
    joblib.dump(rf_model, "../models/random_forest.pkl")
    joblib.dump(xgb_model, "../models/xgboost_best.pkl")
    print("Models saved ✅")
if __name__ == "__main__":
    train_and_save()


# This code trains a Random Forest and XGBoost model on the credit card fraud dataset, evaluates their performance, and saves the models to disk.
# It uses class weighting to handle the imbalanced dataset and prints classification reports and confusion matrices for both models.
# The Random Forest model is trained with all CPU cores for faster training, and the XGBoost model uses a scale_pos_weight to adjust for class imbalance.
# The models are saved in the "../models/" directory.
