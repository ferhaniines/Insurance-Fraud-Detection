# 🔍 Insurance Fraud Detection

---

## 📌 Project Overview

This project explores an auto insurance claims dataset to identify patterns associated with fraudulent claims. The objective is to perform exploratory data analysis (EDA), engineer meaningful features, and highlight variables that can support fraud detection modeling.

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
│   └── 02_Feature_Engineering.ipynb        # Feature engineering & preprocessing
│
└── README.md
```
---

## 🔬 Workflow

### `01_EDA.ipynb` ✅
- Preliminary data inspection (shape, types, nulls, duplicates)
- Data cleaning (null handling, weird values)
- Categorical variable analysis with fraud rate visualisation
- Numerical variable analysis (distributions, correlations, outlier detection)
 
### `02_Feature_Engineering.ipynb` ✅
- Binary encoding and one-hot encoding of categorical variables
- New features: auto age, claim ratios, behavioural flags, etc
- Outlier treatment using IQR
- Feature selection based on correlation with target
- Three feature sets exported for modelling
 
### `03_Modelling.ipynb` 🔄
- Baseline cross-validation across 6 classifiers
- Hyperparameter tuning with GridSearchCV
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Feature importance and model explainability
 
---

## 🔑 Key Findings

- `incident_severity`, `vehicle_claim`, `property_claim` and `total_claim` are the most correlated with target. Higher claims show higher fraud and major damages show 60% fraud rate.

- `authorities_contacted` displays 32% fraud rate for "Other"


---
 
## 📅 Project History
 
This project was originally completed in 2022 as part of a self-directed learning exercise. In 2025, it was revisited and substantially revised to address limitations identified with additional experience: train/test leakage in the ML evaluation, incomplete numerical analysis, and the absence of class imbalance handling. The original notebook is preserved in `Notebooks/Legacy/` for transparency.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine Learning | `sklearn` |
| Environment | Python 3.9+ / Jupyter Notebook |
