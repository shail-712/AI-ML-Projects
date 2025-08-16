# 💳 Credit Card Fraud Detection

> **Machine Learning Classification Pipeline using XGBoost**

A comprehensive fraud detection system that classifies credit card transactions using anonymized data from Kaggle. Built to handle extreme class imbalance with state-of-the-art ML techniques.

## 📊 Dataset Overview

| **Attribute** | **Details** |
|---------------|-------------|
| **Source** | [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| **Size** | 284,807 transactions |
| **Features** | V1–V28 (PCA-transformed) + Time + Amount |
| **Target** | Class (0 = legitimate, 1 = fraud) |
| **Challenge** | Highly imbalanced dataset |

## 🏗️ Project Architecture

```
credit_card_fraud_detection/
├── 📁 ccfd_utils/
│   └── preprocessing.py
├── 📁 notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_modelling.ipynb
├── 📁 models/
│   └── xgboost_best.pkl
├── 🐍 training_model.py
├── 🐍 evaluate_model.py
├── 🐍 predict.py
├── 📋 requirements.txt
└── 📖 README.md
```

## ✨ Features & Implementation

- **📈 Comprehensive EDA** with statistical analysis and visualizations
- **🔧 Advanced Preprocessing** including feature scaling and train/test splits
- **⚖️ Class Imbalance Handling** using `class_weight` and `scale_pos_weight`
- **🤖 Multi-Model Training** (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- **📊 Performance Benchmarking** with detailed model comparison
- **💾 Model Persistence** for production deployment
- **🎯 Real-time Prediction** capability

## 🏆 Model Performance Results

| Model | Precision | Recall | F1-Score | Status |
|-------|:---------:|:------:|:--------:|:------:|
| Logistic Regression | 0.06 | 0.92 | 0.11 | ❌ High false positives |
| Decision Tree | 0.72 | 0.71 | 0.72 | ⚠️ Moderate performance |
| Random Forest | 0.96 | 0.74 | 0.84 | ✅ Strong balance |
| **XGBoost** | **0.89** | **0.84** | **0.86** | 🥇 **Best performer** |

> **🚨 Important:** Accuracy metrics are intentionally excluded due to extreme class imbalance. Precision and recall provide more meaningful insights for fraud detection scenarios.

### 🎯 Key Insights

- **Logistic Regression**: Achieved excellent recall (92%) but suffered from poor precision (6%), generating excessive false positives
- **Decision Tree**: Provided balanced performance with moderate precision and recall
- **Random Forest**: Delivered high precision (96%) with good recall (74%)
- **XGBoost**: Optimal solution with strong recall (84%) and high precision (89%), minimizing both false positives and false negatives

## 🚀 Quick Start Guide

### 1️⃣ Setup Environment
```bash
pip install -r requirements.txt
```

### 2️⃣ Train Model
```bash
python training_model.py
```

### 3️⃣ Evaluate Performance
```bash
python evaluate_model.py
```

### 4️⃣ Make Predictions
```bash
python predict.py
```

## 📦 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
imbalanced-learn>=0.8.0
```

## 🔮 Future Enhancements

- [ ] **Cross-validation** implementation for robust model validation
- [ ] **Threshold optimization** for precision-recall trade-offs
- [ ] **Streamlit dashboard** for interactive fraud detection
- [ ] **API endpoint** for real-time transaction scoring
- [ ] **Model monitoring** and performance tracking
- [ ] **Advanced ensemble methods** exploration

## 📄 License

This project is available under the MIT License.

---

*Built with ❤️ for fraud prevention and financial security*
