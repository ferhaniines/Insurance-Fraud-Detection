# 🔍 Insurance Fraud Detection

This project explores an auto insurance claims dataset to identify patterns associated with fraudulent claims. The objective is to perform exploratory data analysis (EDA), engineer meaningful features, and highlight variables that can support fraud detection modeling.

---

## 📌 Project Overview

Insurance fraud is a significant issue in the industry. According to a CBC News report, fraudulent claims in Canada are estimated to range between **12% and 30%**.  

The fraud rate observed in this dataset (**25%**) falls within this range, suggesting that the dataset provides a **realistic representation** for fraud detection analysis.  

Source: https://www.cbc.ca/news/canada/fraud-prevalent-in-auto-insurance-claims-study-1.291463

**Dataset:** [Insurance Claims Fraud Detection – Kaggle](https://www.kaggle.com/code/meddhiaeddinedabbech/fraud-detection-in-insurance-claims/input?select=insurance_claims.csv)  

**Dataset Description:**
The dataset contains 1,000 insurance claims with a mix of:
- Customer demographics
- Policy details
- Incident characteristics
- Claim amounts
- Fraud label (`fraud_reported`)

---
 
## 📅 Project History
 
This project was originally completed in 2022 as part of a self-directed learning exercise. In 2025, it was revisited and substantially revised to address limitations identified with additional experience: train/test leakage in the ML evaluation, incomplete numerical analysis, and the absence of class imbalance handling. The original notebook is preserved in `Notebooks/Legacy/` for transparency.

---

## 📁 Repository Structure

```
insurance-fraud-detection/
│
├── Dataset/
│   ├── Processed/
│   │   ├── X_tree.csv                      # Features file for tree based modelling
│   │   ├── X_linear_scaled.csv             # Features file for linear modelling
│   │   └── X_dist_scaled.csv               # Features file for distance based modelling
│   │
│   └── insurance_claims.csv                # Raw dataset (download from Kaggle)
│
├── notebooks/
│   ├── Legacy/
│   │   └── Original_Analysis_2023.ipynb    # Original analysis from 2023 archived as is
│   │
│   ├── 01_EDA.ipynb                        # Exploratory Data Analysis & Data Cleaning
│   ├── 02_Feature_Engineering.ipynb        # Feature engineering & preprocessing
│   └── 03_Modelling.ipynb                  # Modelling of processed feature files
│
├── README.md
└── requirements.txt
```
---

## 🔬 Workflow

### `01_EDA.ipynb`
- Preliminary data inspection (shape, types, nulls, duplicates)
- Data cleaning (null handling, weird values, impossible records)
- Categorical variable analysis with fraud rate visualisation
- Numerical variable analysis (distributions, correlations, outlier detection)
 
### `02_Feature_Engineering.ipynb`
- Encoding of categorical variables (binary mapping, one-hot encoding)
- Creation of new features (auto age, claim ratios, behavioural groups, etc)
- Outlier treatment using IQR
- Feature selection based on correlation with target
- Three tailored feature sets for different model families
 
### `03_Modelling.ipynb`
- Stratified train/test split (80/20) with 5-fold cross-validation
- Models compared: Logistic Regression, SVM, KNN, Decision Tree, Random Forest, XGBoost
- Each model trained on its appropriate feature sets (tree-based, linear, distance-based)
- GridSearchCV tuning optimised jointly on ROC-AUC and F1
- Class imbalance handled via `class_weight='balanced'` and `scale_pos_weight`
- **Best model: SVM (linear kernel) — ROC-AUC 0.793, F1 0.638 on fraud class**
 
---

## 🔑 Key Findings

- `incident_severity`, `vehicle_claim`, `property_claim` and `total_claim` are the most correlated with target. Higher claims show higher fraud and major damages show 60% fraud rate.

- `authorities_contacted` displays 32% fraud rate for "Other"

> Fraud patterns are driven by **how a claim is filed**, not by policyholder demographics.

---

## 📈 Results Summary

Six classifiers were evaluated using ROC-AUC and F1 as primary metrics, across three feature sets :
- 71 features : All features except one of each most correlated pairs (corr >= 0.85) without scaling for tree based models only
- 37 features : Top 45 features most correlated with target except one of each most correlated pairs (corr >= 0.7) with scaling
- 28 features : Top 35 features most correlated with target except one of each most correlated pairs (corr >= 0.6) with scaling

Key finding: **fewer, well-selected features consistently outperformed larger sets** with only 980 training rows.

**Test set performance (SVM — best model):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Legitimate | 0.87 | 0.88 | 0.88 |
| Fraud | 0.62 | 0.60 | 0.61 |
| **Overall accuracy** | | | **0.81** |

The model caught **60% of actual fraud cases** (29/48) on the held-out test set.

---

## ⚙️ Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Dataset
Download `insurance_claims.csv` from [Kaggle](https://www.kaggle.com/code/meddhiaeddinedabbech/fraud-detection-in-insurance-claims/input?select=insurance_claims.csv) and place it in the `Dataset/` folder.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine Learning | `sklearn` |
| Environment | Python 3.9+ / Jupyter Notebook |
