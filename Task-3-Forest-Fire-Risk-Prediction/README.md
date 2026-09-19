# Forest Fire Risk Prediction Using Machine Learning

<div align="center">

**AVIRENZA Technologies – AI/ML Internship | Task 3**  
*A Supervised Machine Learning Classification Project*

</div>

---

## Overview

This project builds an end-to-end, reproducible, supervised machine learning classification pipeline to predict historical forest fire occurrences based on environmental, spatial, and meteorological measurements from the official **UCI Forest Fires Dataset**.

The system constructs a leakage-safe Scikit-Learn pipeline featuring robust data preprocessing, exploratory data analysis, class balance verification, model training (**Logistic Regression** and **Random Forest Classifier**), comprehensive cross-validation, hyperparameter-grounded model evaluation, feature importance analysis, and model serialization.

---

## Internship Information

| Attribute | Details |
| :--- | :--- |
| **Intern** | Tharani Natarajan |
| **Institution** | IFET College of Engineering |
| **Department** | Artificial Intelligence and Data Science |
| **Company** | AVIRENZA Technologies |
| **Track** | AI/ML Internship |
| **Task** | Task 3 – Build a Supervised Learning Model (Intermediate) |

---

## Problem Statement

Forest fires present serious environmental and ecological challenges. Meteorological conditions—such as temperature, relative humidity, wind speed, and fuel moisture indices—strongly correlate with fire behavior.

The objective of this project is to model the empirical relationship between observed environmental conditions and historical fire occurrences.

> [!NOTE]
> **Educational Scope:** This project predicts whether historical environmental records were associated with a fire occurrence (`fire_occurrence`). It is designed as an educational machine learning project and is **not** intended as an operational, real-time wildfire warning or evacuation forecasting system.

---

## Dataset & Leakage Prevention

- **Dataset:** UCI Forest Fires Dataset
- **Official Source:** [UCI Machine Learning Repository - Forest Fires](https://archive.ics.uci.edu/dataset/162/forest+fires)
- **Origin:** Montesinho Natural Park, Portugal (Cortez & Morais, 2007)
- **Raw Observations:** 517 rows, 13 attributes
- **Cleaned Dataset:** 513 unique rows (4 exact duplicates removed)

### Target Variable Formulation

The original dataset contains a continuous `area` column representing burned land in hectares. For supervised binary classification, the target is defined as:

$$\text{fire\_occurrence} = \begin{cases} 1, & \text{if } \text{area} > 0 \text{ (Fire / Burned Area Recorded)} \\ 0, & \text{if } \text{area} = 0 \text{ (No Burned Area Recorded)} \end{cases}$$

- **Class 0 (No Burned Area):** 244 instances (47.56%)
- **Class 1 (Burned Area Recorded):** 269 instances (52.44%)
- **Class Balance:** The target is well-balanced (~52.4% vs 47.6%), requiring no synthetic oversampling (SMOTE).

> [!IMPORTANT]
> **Strict Leakage Prevention:** The original `area` variable was **completely excluded** from the feature matrix $X$. Retaining `area` would introduce catastrophic target leakage because `fire_occurrence` is a direct mathematical derivative of `area`.

---

## Technologies Used

- **Python 3.10+ / 3.14**
- **Pandas** & **NumPy** – Data loading, manipulation, and numerical operations
- **Scikit-Learn** – Machine learning pipelines, imputation, scaling, one-hot encoding, modeling, and evaluation
- **Matplotlib** & **Seaborn** – High-resolution statistical visualizations
- **Joblib** – Model serialization
- **Jupyter Notebook** – Interactive experimentation and verification

---

## Preprocessing & Pipeline Architecture

All preprocessing transformations are encapsulated within a `ColumnTransformer` and integrated into Scikit-Learn `Pipeline` objects to prevent data leakage between training and testing folds:

1. **Numerical Features (10):** `['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']`
   - `SimpleImputer(strategy='median')`
   - `StandardScaler()`
2. **Categorical Features (2):** `['month', 'day']`
   - `SimpleImputer(strategy='most_frequent')`
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
3. **Data Splitting:**
   - 80% Training ($N = 410$)
   - 20% Testing ($N = 103$)
   - `stratify=y` with `random_state=42`

```
Input Features (X)
  ├── Numerical Pipeline ──> [Median Imputer] ──> [StandardScaler] ──┐
  │                                                                  ├──> [ColumnTransformer] ──> [Classifier] ──> Prediction
  └── Categorical Pipeline ──> [Mode Imputer]   ──> [OneHotEncoder] ──┘
```

---

## Machine Learning Models & Actual Results

Two supervised classifiers were trained and evaluated on the stratified test set ($N=103$).

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 (Mean ± Std) | 5-Fold CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5437 | 0.5556 | 0.6481 | 0.5983 | 0.6096 | 0.5566 ± 0.0412 | 0.4739 |
| **Random Forest (100 trees)** | **0.6408** | **0.6491** | **0.6852** | **0.6667** | **0.7037** | **0.5522 ± 0.0354** | **0.5398** |

### Confusion Matrix Breakdown (Test Set: N=103)

- **Logistic Regression:**
  - True Negatives (TN): 21 | False Positives (FP): 28
  - False Negatives (FN): 19 | True Positives (TP): 35
- **Random Forest:**
  - True Negatives (TN): 29 | False Positives (FP): 20
  - False Negatives (FN): 17 | True Positives (TP): 37

---

## Feature Importance Analysis

Using `preprocessor.get_feature_names_out()`, the Random Forest model's Gini feature importance was evaluated across all transformed features:

1. **`temp` (Ambient Temperature):** 0.1314
2. **`RH` (Relative Humidity):** 0.1234
3. **`DC` (Drought Code):** 0.0976
4. **`DMC` (Duff Moisture Code):** 0.0959
5. **`ISI` (Initial Spread Index):** 0.0885
6. **`wind` (Wind Speed):** 0.0877
7. **`FFMC` (Fine Fuel Moisture Code):** 0.0760

> [!TIP]
> **Interpretation:** Meteorological variables (temperature and relative humidity) along with moisture indices (DC, DMC, ISI) exhibit the strongest influence on the tree split decisions. This statistical importance reflects model reliance and does **not** prove direct physical causation.

---

## Visualizations Generated

All high-resolution figures are stored under [`outputs/figures/`](file:///outputs/figures/):
- `01_fire_occurrence_distribution.png`: Target class breakdown (47.6% vs 52.4%).
- `02_temperature_distribution.png`: Ambient temperature distribution with mean & median.
- `03_relative_humidity_distribution.png`: Relative humidity distribution.
- `04_fire_occurrence_vs_temperature.png`: Temperature boxplots stratified by fire occurrence.
- `05_correlation_heatmap.png`: Correlation matrix across numerical meteorological indices.
- `06_logistic_regression_confusion_matrix.png`: Logistic Regression confusion matrix.
- `07_random_forest_confusion_matrix.png`: Random Forest confusion matrix.
- `08_roc_curve_comparison.png` / `roc_curve_comparison.png`: ROC Curves comparing both models.
- `09_random_forest_feature_importance.png` / `random_forest_feature_importance.png`: Top 15 Gini feature importances.

---

## Project Structure

```
Task-3-Forest-Fire-Risk-Prediction/
├── README.md                                    <- Comprehensive project documentation
├── data/
│   ├── forestfires.csv                          <- Official UCI dataset file
│   ├── forestfires.names                        <- Attribute descriptions
│   └── README.md                                <- Dataset documentation & leakage rules
├── notebooks/
│   └── forest_fire_prediction.ipynb             <- Complete 25-section reproducible notebook
├── src/
│   └── forest_fire_prediction.py                <- Modular Python script for automated execution
└── outputs/
    ├── figures/                                 <- Visualizations and plots (11 PNGs)
    ├── reports/
    │   ├── model_comparison.csv                 <- Computed performance metrics
    │   ├── model_evaluation.txt                 <- Full text classification report & CV logs
    │   └── viva_questions.md                    <- 20 Viva Q&A for internship defense
    └── models/
        └── random_forest_fire_model.pkl         <- Serialized end-to-end Pipeline (joblib)
```

---

## Limitations

1. **Geographic Scope:** The data originates exclusively from Montesinho Natural Park in Portugal; patterns may not generalize to different biomes or climatic zones.
2. **Sample Size:** With 513 cleaned observations, extreme meteorological events are sparse in the sample.
3. **Derived Binary Target:** Converting continuous burned area into a binary indicator loses nuance regarding fire severity or burned perimeter magnitude.
4. **Non-Operational:** The model is an empirical educational exercise and cannot replace physical fire hazard indices or satellite monitoring systems.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Python Pipeline
```bash
python src/forest_fire_prediction.py
```

### 3. Run Interactive Notebook
```bash
jupyter notebook notebooks/forest_fire_prediction.ipynb
```

---

## Conclusion

This project successfully implements a robust, modular, and leak-free machine learning workflow for forest fire risk classification. Random Forest demonstrated superior discriminatory capacity (ROC-AUC: 0.7037, Accuracy: 64.08%) over Logistic Regression (ROC-AUC: 0.6096, Accuracy: 54.37%), with temperature, humidity, and drought codes emerging as key predictive indicators.
