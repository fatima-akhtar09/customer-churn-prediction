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
* **Top features by permutation importance:** **tenure**, **TotalCharges**, **Contract_Two year**
* **Threshold chosen:** **0.30**, because it increases recall and helps catch more potential churners when missing a churner is costly.
* **Engineered features:** **n_services, is_new, charge_per_mo, price_jump**; effect on AUC: **0.8422 → 0.8420**
* **Biggest lesson:** **Model evaluation should consider the business cost of false negatives and false positives, not accuracy alone.**

  
## Week 3: Model Optimization and Unsupervised Learning

### Model Optimization

* **20-seed accuracy range:** 0.780 to 0.828
* **5-fold CV AUC:** LR **0.8464 ± 0.0129**, RF **0.8464 ± 0.0114**, XGBoost **0.8504 ± 0.0125**
* **Best Random Forest parameters:** `max_depth=15, max_features=0.2127, min_samples_leaf=15`
* **Grid Search vs Random Search:** **179s vs 133s**

### Final Model

**XGBoost (tuned)** was selected based on the highest cross-validated AUC (**0.8504**). On the held-out test set, it achieved a **test AUC of 0.8478**.

### Customer Segmentation

K-Means with **k = 4** produced four customer segments with different observed churn rates:

| Segment                   | Churn Rate |
| ------------------------- | ---------: |
| Mid-tenure, high-spend    |        43% |
| New, low-service          |        32% |
| Long-tenure, high-service |        14% |
| Long-tenure, low-spend    |         5% |

### PCA

**15 of 30 principal components** were required to explain at least **90% of the variance**, indicating redundancy among some of the original features.

### Biggest Lesson

**Cross-validation provides a reliable basis for model selection, while clustering and PCA reveal useful customer segments and redundancy in the feature set.**

