# Customer Churn Prediction

## Week 1: Exploratory Data Analysis

### Dataset
- Source: Telco Customer Churn (Kaggle)
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)

### Key Findings
- The dataset contains **7,043 customers** across **21 features**.
- The overall customer churn rate is **26.54%**, with **1,869 customers** having churned.
- There are **11 missing values** in the `TotalCharges` column.
- The dataset contains **demographic, service, and account-related information** that can be used to analyze customer churn.

### Setup
Open the Kaggle notebook or run locally:

pip install pandas numpy matplotlib seaborn

## Week 2: Building ML Models

* **Baseline (always "stay"):** accuracy **0.735**
* **Best model:** Logistic Regression, AUC **0.842**, recall **0.754** at threshold **0.30**
* **Top churn drivers (permutation importance):** **tenure**, **TotalCharges**, **Contract_Two year**
* **Threshold chosen:** **0.30**, because it increases recall and helps catch more potential churners when missing a churner is costly.
* **Engineered features:** **n_services, is_new, charge_per_mo, price_jump**; effect on AUC: **0.8422 → 0.8420**
* **Biggest lesson:** **Model evaluation should consider the business cost of false negatives and false positives, not accuracy alone.**

