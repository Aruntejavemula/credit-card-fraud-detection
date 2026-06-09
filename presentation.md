# Credit Card Fraud Detection

### ML Pipeline for Identifying Fraudulent Transactions with Class Imbalance Handling

**Author:** Arun Teja Vemula  
**Dataset:** Kaggle — Credit Card Fraud Detection (ULB Machine Learning Group)  
**Date:** June 2026

---

## Introduction

Banking fraud poses a massive financial threat — estimated to cost over $30 billion globally by 2020 (Nilson Report). Every fraudulent transaction is a direct loss for the bank, and the challenge is detecting that fraud automatically, in real-time, without flagging thousands of legitimate transactions.

This project builds a complete machine learning pipeline to **classify credit card transactions as fraudulent or legitimate** from anonymized transaction features — using multiple ML algorithms with a focused approach to handling extreme class imbalance.

---

## The Problem: Extreme Class Imbalance

- **Dataset:** 284,807 transactions over 2 days (Sept 2013, European cardholders)
- **Fraudulent transactions:** Only 492 — just **0.172%** of the data
- **Features:** V1–V28 (PCA-anonymized for privacy), `Time`, `Amount`, `Class`
- **Challenge:** Standard accuracy is meaningless — a model predicting "no fraud" every time scores 99.83% accuracy but catches zero fraud

| Metric | Value |
|--------|-------|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.172%) |
| Legitimate | 284,315 (99.828%) |
| Features | 30 (V1–V28 + Time + Amount) |
| Missing values | None |

---

## Why Standard Models Fail on Imbalanced Data

A naive classifier trained on this data will simply learn to predict "not fraud" every time — this gets 99.8% accuracy but **0% recall on fraud**.

**Metric choice:** ROC-AUC score — measures the model's ability to rank fraud above legitimate transactions regardless of threshold. Better than accuracy for imbalanced classification.

**Techniques tried:**
- **Raw imbalanced data** — train models as-is, use threshold tuning
- **Undersampling** — RandomUnderSampler to balance classes by downsampling majority
- **Oversampling** — SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic fraud samples using nearest neighbors

---

## Methodology

### Phase 1 — Data Preparation
- Train/test split (stratified to preserve class ratio)
- `StandardScaler` on `Amount` and `Time` (V1–V28 already PCA-scaled)
- `PowerTransformer` to correct skewness in features

### Phase 2 — Outlier Treatment
- Identify outliers using IQR method on fraud class
- Remove extreme outliers to reduce noise in the minority class

### Phase 3 — Model Building

**On imbalanced data (raw):**

| Model | Approach |
|-------|----------|
| Logistic Regression | GridSearchCV, KFold=5, tuned C parameter |
| XGBoost | Gradient boosted trees, handles imbalance natively |
| Decision Tree | Full tree with pruning via max_depth |
| Random Forest | Ensemble of decision trees |

**On balanced data (Undersampling):**
- RandomUnderSampler → Logistic Regression

**On balanced data (SMOTE):**
- SMOTE oversampling → Logistic Regression
- Adasyn (adaptive density oversampling) noted as variant

### Phase 4 — Threshold Tuning
- Plot FPR vs TPR curve
- Select optimal threshold — maximizes recall (catching fraud) while keeping FPR acceptable

---

## Results

### Imbalanced Data — Model Comparison

| Model | Train ROC-AUC | Test ROC-AUC |
|-------|:---:|:---:|
| Logistic Regression | ~0.98 | ~0.97 |
| **XGBoost** | **1.00** | **0.98** |
| Decision Tree | ~0.95 | ~0.93 |
| Random Forest | ~0.99 | ~0.97 |

**Best model: XGBoost — Test ROC-AUC 0.98**

### Why XGBoost Won
- ROC-AUC of 0.98 on test data — highest of all models
- Logistic Regression was close (0.97) but XGBoost's 0.01 edge translates to significant real savings in a banking context
- Caveat: XGBoost is more resource-intensive than Logistic Regression

---

## Key Findings

1. **XGBoost is the best model** — Test ROC-AUC 0.98, narrowly beating Logistic Regression (0.97) and Random Forest (0.97). The small margin matters at banking scale.

2. **Threshold tuning is critical** — Default 0.5 threshold is wrong for fraud. Shifting the threshold toward lower values dramatically increases recall at the cost of more false positives. Business context determines the right trade-off.

3. **SMOTE outperforms undersampling in practice** — Undersampling discards 99%+ of data; SMOTE generates synthetic minority samples, preserving information. Better generalization on test set.

4. **Feature V14 is the most important predictor** — XGBoost feature importance shows V14 (a PCA-anonymized component) contributes most to fraud prediction, followed by V4, V10, V12.

5. **KNN was excluded** — Not memory-efficient at 284K rows; too slow for production.

6. **SVM was excluded** — Computationally expensive on this scale; not practical without heavy kernel tuning.

---

## Summary

This project delivered a complete end-to-end ML fraud detection pipeline:

- Processed 284,807 transactions across 30 features with no missing values
- Handled severe class imbalance (0.172% fraud) using undersampling and SMOTE
- Evaluated Logistic Regression, XGBoost, Decision Tree, and Random Forest
- **Best result: XGBoost with ROC-AUC 0.98** on held-out test data
- Applied threshold optimization to maximize fraud recall for real-world deployment

---

## Conclusion

Credit card fraud detection is a solved problem at the research level but a challenging engineering problem in practice. The critical insight is that **accuracy is the wrong metric** — ROC-AUC and recall are what matter. Fine-tuning the decision threshold is equally important as model selection.

**Future work:**
- Neural network approach (LSTM for sequential transaction patterns)
- Cost-sensitive learning (penalize missed fraud heavier than false positives)
- Real-time streaming inference pipeline (Kafka + model server)
- Ensemble of XGBoost + Logistic Regression for robustness

---

## References

*(APA 7th Edition)*

Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE Symposium Series on Computational Intelligence*, 159–166.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794. https://doi.org/10.1145/2939672.2939785

Machine Learning Group — ULB. (2018). *Credit card fraud detection* [Dataset]. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Nilson Report. (2019). *Card fraud losses worldwide*. HSN Consultants, Inc.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

---

*For the full PowerPoint presentation, see `Credit_Card_Fraud_Detection_Presentation.pptx`*
