# Credit Card Fraud Detection

> ML pipeline for detecting fraudulent credit card transactions with class imbalance handling
> **Best ROC-AUC: 0.98** (XGBoost on imbalanced data)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-0.98_ROC--AUC-green.svg)](https://xgboost.readthedocs.io)
[![Kaggle](https://img.shields.io/badge/Kaggle-ULB_Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Overview

Banking fraud is estimated to cost over **$30 billion globally** per year. This project builds a complete machine learning pipeline to **automatically classify credit card transactions as fraudulent or legitimate** — evaluating multiple algorithms and addressing the core challenge: extreme class imbalance (only 0.172% of transactions are fraud).

---

## The Challenge: Extreme Class Imbalance

| Metric | Value |
|--------|-------|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.172%) |
| Legitimate | 284,315 (99.828%) |
| Features | 30 (V1–V28 PCA + Time + Amount) |
| Missing values | None |

A naive model that predicts "no fraud" every time scores **99.83% accuracy but catches zero fraud**. Standard accuracy is useless here — this project uses **ROC-AUC** as the primary metric.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group)
- **Period:** 2 days in September 2013, European cardholders
- **Features:** V1–V28 are PCA-anonymized (privacy), plus `Time`, `Amount`, and `Class` (1=fraud, 0=legit)
- **Note:** Dataset file (`creditcard.csv`, ~144MB) not included — download from Kaggle

---

## Methodology

### Phase 1 — Data Preparation
- Train/test split (stratified to preserve class ratio)
- `StandardScaler` on `Amount` and `Time` (V1–V28 already PCA-scaled)
- `PowerTransformer` to correct feature skewness
- IQR-based outlier treatment on fraud class

### Phase 2 — Model Building on Imbalanced Data

| Model | Notes |
|-------|-------|
| **Logistic Regression** | GridSearchCV, KFold=5, tuned C parameter, ROC-AUC scoring |
| **XGBoost** | Gradient boosted trees; handles imbalance natively via scale_pos_weight |
| **Decision Tree** | Pruned via max_depth; interpretable baseline |
| **Random Forest** | Ensemble baseline; high resource cost |

### Phase 3 — Handling Class Imbalance

**Undersampling:** RandomUnderSampler reduces 284,315 legitimate samples to 492 to match fraud count. Fast but discards 99.8% of data.

**SMOTE:** Synthetic Minority Over-sampling Technique generates new synthetic fraud samples using K-Nearest Neighbors. Preserves all original data — better generalization than undersampling.

### Phase 4 — Threshold Optimization
- Plot FPR vs TPR across all thresholds
- Select optimal threshold to maximize fraud recall
- Default 0.5 threshold is wrong for fraud — lowering it increases recall at cost of more false positives

---

## Results

| Model | Data | Train ROC-AUC | Test ROC-AUC |
|-------|------|:---:|:---:|
| Logistic Regression | Imbalanced | ~0.98 | ~0.97 |
| **XGBoost** | **Imbalanced** | **1.00** | **0.98** |
| Decision Tree | Imbalanced | ~0.97 | ~0.93 |
| Random Forest | Imbalanced | ~0.99 | ~0.97 |
| LR + Undersampling | Balanced (RUS) | ~0.95 | ~0.94 |
| LR + SMOTE | Balanced (SMOTE) | ~0.97 | ~0.96 |

**Best model: XGBoost — Test ROC-AUC 0.98**

---

## Key Findings

1. **XGBoost wins** — Test ROC-AUC 0.98, 0.01 ahead of Logistic Regression. That margin translates to real savings at banking scale.
2. **Accuracy is the wrong metric** — ROC-AUC correctly handles severely imbalanced classification.
3. **Threshold matters** — Optimal decision threshold is a business decision, not a technical default.
4. **SMOTE beats undersampling** — Undersampling discards 99.8% of data; SMOTE generates synthetic samples while keeping everything.
5. **V14 is the top predictor** — XGBoost feature importance shows V14 (anonymized PCA component) contributes most to fraud detection.
6. **KNN/SVM excluded** — KNN is O(n) at inference time (284K rows is too slow); SVM requires expensive kernel tuning at this scale.

---

## Repository Contents

| File | Description |
|------|-------------|
| `credit_card_fraud_detection.ipynb` | Main notebook — full pipeline with saved outputs |
| `Solution-Approach.pdf` | Written solution approach document |
| `credit-card-fraud-detection-main.zip` | Zipped source archive |
| `presentation.md` | Presentation-style summary |

---

## Environment

```
Python        3.x
scikit-learn  (sklearn)
xgboost
imbalanced-learn  (imblearn)
pandas, numpy, matplotlib, seaborn
```

**Data file** (not included — download from Kaggle):
- `creditcard.csv` — 284,807 transactions, ~144MB

---

## Why Not SVM or KNN?

**KNN:** Not memory-efficient at this scale — the algorithm stores all training data and computes distances for every prediction. With 284K rows, inference becomes prohibitively slow.

**SVM:** Computationally expensive kernel optimization at this data size. Not practical without significant infrastructure.

---

## References

Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR, 16*, 321–357.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*. https://doi.org/10.1145/2939672.2939785

Machine Learning Group — ULB. (2018). *Credit card fraud detection* [Dataset]. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Nilson Report. (2019). *Card fraud losses worldwide*. HSN Consultants, Inc.

---

*Author: Arun Teja Vemula*
