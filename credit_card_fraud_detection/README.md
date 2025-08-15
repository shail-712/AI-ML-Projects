| Model               | Precision | Recall   | F1-score |
| ------------------- | --------- | -------- | -------- |
| Logistic Regression | 0.06      | 0.92     | 0.11     |
| Decision Tree       | 0.72      | 0.71     | 0.72     |
| Random Forest       | 0.96      | 0.74     | 0.84     |
| **XGBoost**         | **0.89**  | **0.84** | **0.86** |


🚨 Note: Accuracy is not included here because the dataset is highly imbalanced — metrics like recall and precision are more relevant for fraud detection.


**Conclusion:**
The baseline Logistic Regression achieved very high recall but extremely poor precision, producing too many false positives. Decision Tree improved both metrics, while Random Forest provided a strong balance. XGBoost achieved the best overall performance, with a high recall (0.84) and relatively high precision (0.89), making it the most effective model for this dataset.