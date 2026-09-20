# Task 4 – Feature Engineering & Model Improvement: Electricity Consumption Prediction

**AVIRENZA Technologies – AI/ML Internship**  
**Student:** Tharani Natarajan  
**College:** IFET College of Engineering  
**Department:** Artificial Intelligence and Data Science  
**Project Type:** Supervised Machine Learning – Regression  

---

## 1. Executive Summary

This project delivers a complete, leak-free, production-grade machine learning system demonstrating how **domain-specific feature engineering** and **time-aware model tuning** transform a weak predictive baseline into a high-precision forecasting model.

Using 47 months of hourly electrical load measurements from the official **UCI Individual Household Electric Power Consumption** repository (34,421 processed hourly observations), we engineered 26 temporal, cyclical, autocorrelation lag, and historical rolling window features. By replacing a naive linear baseline with a feature-engineered Random Forest regressor, predictive error dropped by **40.8% (MAE reduced from 0.5179 kW to 0.3063 kW)**, and the variance explained ($R^2$) increased by **173% (from 0.2299 to 0.6284)**.

---

## 2. Project Architecture & Repository Layout

```
Task-4-Electricity-Consumption-Prediction/
├── data/
│   ├── household_power_consumption_hourly.csv    # Cleaned hourly dataset (34,589 records)
│   └── README.md                                 # Dataset lineage, schema, and preprocessing details
├── notebooks/
│   └── electricity_consumption_prediction.ipynb  # Interactive, executable Jupyter Notebook
├── outputs/
│   ├── figures/                                  # 9 High-resolution evaluation visualizations
│   │   ├── 01_consumption_over_time.png
│   │   ├── 02_avg_consumption_by_hour.png
│   │   ├── 03_avg_consumption_by_day_of_week.png
│   │   ├── 04_avg_consumption_by_month.png
│   │   ├── 05_actual_vs_predicted_time_series.png
│   │   ├── 06_actual_vs_predicted_scatter.png
│   │   ├── 07_residual_error_distribution.png
│   │   ├── 08_baseline_vs_improved_comparison.png
│   │   └── 09_feature_importance.png
│   ├── models/
│   │   └── electricity_consumption_model.pkl     # Serialized tuned Random Forest model
│   └── reports/
│       ├── model_comparison.csv                  # Quantitative baseline vs improved comparison table
│       ├── feature_engineering_summary.csv       # Taxonomy of all 26 engineered features
│       ├── error_analysis.txt                    # Regime-based error diagnosis (peak, diurnal, weekend)
│       └── 08_baseline_vs_improved_comparison.png
├── src/
│   └── electricity_consumption_prediction.py     # Modular, end-to-end Python pipeline
├── README.md                                     # Main project documentation
├── viva_questions.md                             # 15 In-depth interview/viva preparation Q&As
└── demo_video_script.md                          # Professional 3-5 minute demo presentation script
```

---

## 3. Dataset Description & Preprocessing

- **Dataset:** UCI Individual Household Electric Power Consumption
- **Observation Period:** December 16, 2006 to November 26, 2010 (~47 months)
- **Original Frequency:** 1-minute sampling (2,075,259 records)
- **Downsampling / Aggregation:** Resampled to 1-hour intervals (`mean()` of active power, reactive power, voltage, intensity, and sub-meterings).
- **Missing Value Handling:** Handled missing values through linear interpolation across continuous timestamps.
- **Target Variable:** `Global_active_power` (household global active power in kilowatts [kW]).

---

## 4. Advanced Feature Engineering & Leakage Prevention

A central focus of this project is demonstrating **leakage-free feature design**:

### 4.1 Strict Leakage Prevention Rules
1. **Chronological Splitting:** The data is split strictly chronologically:
   - **Training Set:** First 80% of data (Dec 23, 2006 to Feb 13, 2010 — 27,536 samples)
   - **Test Set:** Subsequent 20% of data (Feb 13, 2010 to Nov 26, 2010 — 6,885 samples)
   - *No random shuffling* is permitted.
2. **Lagged Windowing with `shift(1)`:** All rolling statistics (24h mean, min, max, std) are computed strictly on `target.shift(1)` to ensure the current hour's power consumption is never visible when calculating historical summary statistics.

### 4.2 Feature Taxonomy (26 Features)

| Category | Features | Description |
|---|---|---|
| **Temporal (7)** | `hour`, `day_of_week`, `month`, `day_of_year`, `quarter`, `is_weekend`, `is_peak_hour` | Captures calendar patterns and discretionary evening load spikes (18:00–22:00). |
| **Cyclical (6)** | `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos` | Enforces mathematical continuity across cyclic boundaries (e.g. 23:00 to 00:00). |
| **Historical Lags (6)** | `lag_1`, `lag_2`, `lag_3`, `lag_24`, `lag_48`, `lag_168` | Direct autoregressive target dependencies (past 1-3 hours, previous day, 2 days prior, and same hour last week). |
| **Rolling Windows (5)** | `rolling_mean_24`, `rolling_std_24`, `rolling_min_24`, `rolling_max_24`, `rolling_mean_168` | Moving summary statistics over recent 24-hour and 7-day (168h) windows computed on `shift(1)`. |
| **Auxiliary Electrical (2)**| `Voltage`, `Global_reactive_power` | Concurrent electrical grid measurements. |

---

## 5. Experimental Results & Model Comparison

Models were evaluated on the held-out 6,885 test samples using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$), alongside 5-split `TimeSeriesSplit` cross-validation:

| Model | Features | Test MAE (kW) | Test RMSE (kW) | Test $R^2$ | CV MAE (kW) | CV RMSE (kW) | CV $R^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Linear Regression** | 5 | 0.5179 | 0.6382 | 0.2299 | 0.6244 | 0.8015 | 0.2084 |
| **Improved Random Forest (FE)**| 26 | 0.3064 | 0.4443 | 0.6267 | 0.3515 | 0.5037 | 0.6857 |
| **Tuned Random Forest** | 26 | **0.3063** | **0.4433** | **0.6284** | **0.3512** | **0.5020** | **0.6879** |

### Key Improvements:
- **MAE Reduction:** Improved by **40.8%** over baseline.
- **RMSE Reduction:** Improved by **30.5%** over baseline.
- **Variance Explained ($R^2$):** Increased from **0.23** to **0.63** (+173% relative improvement).
- **Generalization:** Time-series cross-validation confirms high stability with CV $R^2$ reaching **0.6879**.

---

## 6. Top Feature Importances

Tree impurity reduction (Gini importance) analysis reveals the most influential predictors:
1. `lag_1` (Immediate 1-hour prior consumption) — **Highest individual predictor** (~34% importance).
2. `rolling_mean_24` (24-hour historical moving average) — Captures baseline household consumption level.
3. `lag_2` & `lag_3` (Short-term consumption momentum).
4. `rolling_max_24` & `rolling_std_24` (Recent volatility and appliance surge capacity).
5. `hour_sin` / `hour_cos` (Circadian lifestyle routines).

---

## 7. Error Diagnosis & Insights

Analysis of prediction residuals revealed clear behavioral patterns:
- **Off-Peak Hours (01:00–06:00):** Lowest MAE (~0.18 kW). Basal refrigeration and standby consumption are highly regular.
- **Peak Evening Hours (18:00–22:00):** Higher MAE (~0.44 kW). Erratic high-draw appliances (cooking stoves, washing machines, electric heaters) create stochastic spikes that are harder to predict from past load alone.
- **Weekdays vs. Weekends:** Weekend load profile exhibits shifted morning wake-up hours and midday activity variance.

---

## 8. How to Run & Reproduce

### 1. Run Complete Python Pipeline
```bash
python src/electricity_consumption_prediction.py
```
*Outputs generated:* Trained model in `outputs/models/`, 9 plots in `outputs/figures/`, and reports in `outputs/reports/`.

### 2. Run Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/electricity_consumption_prediction.ipynb
```

---

## 9. Author & Verification

- **Author:** Tharani Natarajan
- **Institution:** IFET College of Engineering, Department of Artificial Intelligence and Data Science
- **Internship Organization:** AVIRENZA Technologies
- **Status:** Complete, Verified, Internship-Ready
