# Bank Marketing Data Preprocessing Using Python

<div align="center">

**Task 2 — Build a Data Preprocessing Pipeline**  
**AVIRENZA Technologies | AI/ML Internship**

</div>

---

## Overview

This project delivers a complete, modular, and leak-free **Data Preprocessing Pipeline** for the **UCI Bank Marketing Dataset** using Python and Scikit-Learn. The pipeline automates the ingestion, quality inspection, missing/unknown category handling, outlier auditing, numerical standardization, and categorical one-hot encoding necessary to transform raw banking campaign records into high-quality feature matrices ready for machine learning algorithms.

---

## Internship Details

| Field | Details |
|---|---|
| **Company** | AVIRENZA Technologies |
| **Program** | AI/ML Internship |
| **Task** | Task 2 – Build a Data Preprocessing Pipeline |
| **Student** | Tharani Natarajan |
| **Institution** | IFET College of Engineering |
| **Department** | Artificial Intelligence and Data Science |

---

## Objective

The objective of this project is to prepare real-world bank marketing campaign data for predictive machine learning models by establishing a robust, reproducible, and leak-free preprocessing pipeline using Scikit-Learn's `Pipeline` and `ColumnTransformer`.

---

## Dataset

- **Dataset Name:** UCI Bank Marketing Dataset
- **Primary Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) (Moro, Cortez, & Rita, 2014)
- **Dataset Purpose:** Direct phone marketing campaign records of a Portuguese banking institution to predict whether a client will subscribe to a bank term deposit.
- **Dataset Dimensions:** **45,211 rows** and **17 columns** (16 input features + 1 target).
- **Target Variable:** `y` (`yes` = subscribed, `no` = did not subscribe).
- **Class Balance:** Imbalanced target — `no`: 39,922 (88.30%), `yes`: 5,289 (11.70%).
- **File Used:** `data/bank-full.csv` (semicolon-delimited `;`).

---

## Technologies Used

- **Python 3.x**
- **Pandas & NumPy** — Data manipulation and matrix operations
- **Scikit-Learn** — Preprocessing pipelines (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `train_test_split`)
- **Matplotlib & Seaborn** — Publication-quality visualizations
- **Jupyter Notebook** — Interactive exploratory workflows

---

## Preprocessing Steps

1. **Data Ingestion & Delimiter Detection:** Automatic handling of semicolon-separated CSV files.
2. **Initial Data Quality Auditing:** Comprehensive checks for missing values (`NaN`), duplicates, and data types across all 17 features.
3. **Domain "Unknown" Value Strategy:** Retention and distinct encoding of domain `"unknown"` entries (e.g., `poutcome`: 81.75%, `contact`: 28.80%) which represent critical operational context (e.g., uncontacted clients).
4. **Duplicate Record Handling:** Verified dataset integrity (0 duplicate rows found).
5. **Feature Taxonomy Identification:** Automated partitioning into 7 numerical and 9 categorical attributes.
6. **Target Variable Encoding:** Binary mapping of `y` (`no` $\rightarrow 0$, `yes` $\rightarrow 1$).
7. **Stratified Train-Test Splitting (80/20):** Strict isolation of test data before transformation to eliminate data leakage.
8. **Numerical Transformation Sub-Pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
9. **Categorical Transformation Sub-Pipeline:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
10. **Unified Composite Transformer:** Integrated `ColumnTransformer` producing 51 encoded features.
11. **Transformation & Validation:** Zero missing values, correct unit-variance scaling, and verified baseline classification readiness.

---

## Data Leakage Prevention

Data leakage occurs when information from outside the training dataset (such as mean/standard deviation or test set category frequencies) is inadvertently used to train or fit preprocessing models. 

In this project, data leakage is strictly prevented:
- The dataset is split into `X_train` and `X_test` **before** any transformations are fitted.
- The `ColumnTransformer` is **fitted exclusively on `X_train`** (`fit_transform`).
- The testing set `X_test` is transformed using **already-fitted parameters** (`transform`) without recalculating means, standard deviations, or category vocabularies.

---

## Visualizations Generated

All figures are saved in high resolution under `outputs/figures/`:

| Figure | File Name | Description |
|---|---|---|
| **01** | `01_target_distribution.png` | Distribution of deposit subscription target variable (`y`) |
| **02** | `02_missing_unknown_analysis.png` | Percentage breakdown of `"unknown"` entries across categorical features |
| **03** | `03_numerical_distributions.png` | Histograms and KDE curves for key numerical attributes |
| **04** | `04_correlation_heatmap.png` | Upper-triangle correlation heatmap of numerical features |
| **05** | `05_categorical_features_vs_target.png` | Stacked bar charts showing subscription rates across categorical levels |
| **06** | `06_outlier_boxplots.png` | Boxplots analyzing outlier spreads for numerical variables |
| **07** | `07_preprocessed_feature_distributions.png` | Verification of zero-mean, unit-variance standardized distributions |

---

## Project Structure

```
Task-2-Bank-Marketing-Data-Preprocessing/
├── README.md                                <- Project documentation (this file)
├── data/
│   ├── README.md                            <- Dataset documentation & source info
│   ├── bank-full.csv                        <- Complete dataset (45,211 rows)
│   ├── bank.csv                             <- Sample dataset (4,521 rows)
│   └── bank-names.txt                       <- Original UCI attribute metadata
├── notebooks/
│   └── bank_marketing_preprocessing.ipynb   <- Step-by-step executed Jupyter Notebook
├── src/
│   └── bank_marketing_preprocessing.py      <- Modular preprocessing pipeline script
└── outputs/
    ├── figures/                             <- Publication-quality visualization plots
    │   ├── 01_target_distribution.png
    │   ├── 02_missing_unknown_analysis.png
    │   ├── 03_numerical_distributions.png
    │   ├── 04_correlation_heatmap.png
    │   ├── 05_categorical_features_vs_target.png
    │   ├── 06_outlier_boxplots.png
    │   └── 07_preprocessed_feature_distributions.png
    └── reports/                             <- Summary reports & viva preparation
        ├── preprocessing_report.txt
        ├── preprocessing_summary.csv
        └── viva_questions.md
```

---

## How to Run

### 1. Install Dependencies
From the repository root:
```bash
pip install -r requirements.txt
```

### 2. Run the Modular Preprocessing Script
```bash
python Task-2-Bank-Marketing-Data-Preprocessing/src/bank_marketing_preprocessing.py
```

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook Task-2-Bank-Marketing-Data-Preprocessing/notebooks/bank_marketing_preprocessing.ipynb
```

---

## Results & Validation

- **Raw Instances:** 45,211 rows $\times$ 17 columns
- **Training Samples:** 36,168 rows (80%)
- **Testing Samples:** 9,043 rows (20%)
- **Processed Features:** **51 columns** (7 numerical + 44 one-hot encoded categories)
- **Post-Transformation Missing Values:** 0
- **Baseline Logistic Regression Validation:**
  - Training Accuracy: **90.20%**
  - Testing Accuracy: **90.12%**
  - Demonstrates that preprocessed matrices are fully functional and ML-ready.

---

## Key Learning Outcomes

1. **Pipeline Modularity:** Designed maintainable data pipelines chaining imputers, scalers, and encoders.
2. **Leakage Prevention:** Mastered proper train-test isolation protocols for production machine learning.
3. **Domain Strategy:** Developed nuanced approaches to domain-specific categorical states (`"unknown"`) rather than blind deletion.
4. **Scikit-Learn Mastery:** Applied `ColumnTransformer`, `Pipeline`, `StandardScaler`, and `OneHotEncoder` best practices.

---

## Disclaimer

This project is prepared strictly for educational and internship evaluation purposes as part of the AVIRENZA Technologies AI/ML Internship. It is not a financial decision-making tool and does not offer financial or investment advice.
