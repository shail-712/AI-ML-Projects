# Titanic Survival Prediction 🚢

This project uses machine learning to predict which passengers survived the Titanic disaster based on features like age, gender, class, and more.

## 🔍 Dataset
From [Kaggle Titanic Challenge](https://www.kaggle.com/competitions/titanic)

- `train.csv`: Training data with labels
- `test.csv`: Test data for prediction

## 💡 What You’ll Learn
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building (Logistic Regression, Decision Tree, etc.)
- Model Evaluation
- Hyperparameter Tuning

## ⚙️ Requirements
Install using:
```bash
pip install -r requirements.txt
```





# 🛳️ Titanic Classification Project

## 📌 Objective

Predict passenger survival on the Titanic using:

* **Logistic Regression**
* **Decision Tree Classifier**

---

## 📊 Dataset

* Features: `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch`, `Embarked`, etc.
* Target: `Survived` (0 = No, 1 = Yes)

---

## ⚙️ Models Used

### ✅ Logistic Regression

* **Used for binary classification**
* Predicts probability using sigmoid function:

  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$
* Predicts class 1 if probability ≥ 0.5

Access model parameters:

```python
lr_model.coef_       # Weights
lr_model.intercept_  # Bias
```

---

### 🌳 Decision Tree Classifier

* Splits data based on feature conditions (like `Sex`, `Pclass`, etc.)
* No need for feature scaling
* Handles non-linearity well

Visualize the tree:

```python
from sklearn.tree import plot_tree
plot_tree(dt_model, feature_names=X.columns, filled=True)
```

---

## 🧪 Model Evaluation

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Logistic Regression
y_pred_lr = lr_model.predict(X_val)
print("LR Accuracy:", accuracy_score(y_val, y_pred_lr))

# Decision Tree
y_pred_dt = dt_model.predict(X_val)
print("DT Accuracy:", accuracy_score(y_val, y_pred_dt))
```

---

## 📌 Summary

| Metric                 | Logistic Regression | Decision Tree |
| ---------------------- | ------------------- | ------------- |
| Handles non-linearity  | ❌                   | ✅             |
| Prone to overfitting   | ❌ (Low)             | ✅ (High)      |
| Feature scaling needed | ✅                   | ❌             |

---

