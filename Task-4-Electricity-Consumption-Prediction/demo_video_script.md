# Demo Video Presentation Script (3–5 Minutes)

**Task 4:** Electricity Consumption Prediction with Feature Engineering and Model Improvement  
**Student:** Tharani Natarajan | IFET College of Engineering  
**Department:** Artificial Intelligence and Data Science  
**Internship:** AVIRENZA Technologies  

---

### [00:00 – 00:35] Scene 1: Introduction & Project Overview
**Visual:** Show repository root, project title, student details, and folder hierarchy.
> **Speaker:**  
> "Hello everyone. My name is Tharani Natarajan, studying Artificial Intelligence and Data Science at IFET College of Engineering.  
> Today, I am presenting Task 4 of my AI/ML Internship at AVIRENZA Technologies.  
> The core focus of Task 4 is **Feature Engineering and Model Improvement**.  
> In real-world machine learning, raw data is rarely enough. Our goal in this project is to take 4 years of continuous electricity consumption data from the UCI repository, formulate a weak baseline model, and systematically engineer temporal, cyclical, lag, and rolling features to demonstrate how intelligent feature design dramatically elevates forecasting precision without data leakage."

---

### [00:35 – 01:20] Scene 2: Dataset, Leakage Prevention & Resampling
**Visual:** Open `notebooks/electricity_consumption_prediction.ipynb` showing Section 4 and Section 5 EDA plots (`01_consumption_over_time.png` and `02_avg_consumption_by_hour.png`).
> **Speaker:**  
> "We utilized the official UCI Individual Household Electric Power Consumption dataset, covering 47 months from December 2006 to November 2010.  
> We resampled over 2 million minute-level records into 34,589 stable hourly observations.  
> Because this is a time-series forecasting problem, standard random shuffling causes catastrophic lookahead data leakage.  
> To guarantee zero leakage:  
> First, we enforce strict chronological train-test splitting: the first 80% represents historical training data, and the subsequent 20% represents unseen future testing data.  
> Second, all historical rolling window statistics are strictly calculated on `target.shift(1)` so the current hour's load is never used to predict itself."

---

### [01:20 – 02:15] Scene 3: Feature Engineering Taxonomy
**Visual:** Scroll to Section 6 of the notebook or show `outputs/reports/feature_engineering_summary.csv`.
> **Speaker:**  
> "We engineered 26 comprehensive features across four distinct categories:  
> 1. **Temporal Features:** Extracting calendar indicators, weekend flags, and an empirical peak-hour indicator for the 6 PM to 10 PM load window.  
> 2. **Cyclical Sine/Cosine Encodings:** Transforming hour, day of the week, and month using sine and cosine functions. This ensures mathematical continuity so that hour 23 and hour 0 are treated as adjacent points on a circle rather than distant numbers.  
> 3. **Autoregressive Lags:** Capturing past consumption at 1, 2, 3 hours prior for short-term momentum, 24 and 48 hours for diurnal daily rhythm, and 168 hours for weekly seasonality.  
> 4. **Rolling Window Statistics:** 24-hour moving mean, standard deviation, minimum, and maximum, along with a 7-day rolling average."

---

### [02:15 – 03:15] Scene 4: Baseline vs. Improved Models & Performance Results
**Visual:** Display `outputs/figures/08_baseline_vs_improved_comparison.png` and `outputs/figures/05_actual_vs_predicted_time_series.png`.
> **Speaker:**  
> "Let's look at the results.  
> We established a baseline using Linear Regression trained on 5 basic features. It achieved an MAE of 0.5179 kW and an R-squared of only 0.2299.  
> Next, we trained our improved Random Forest model incorporating all 26 engineered features and performed `TimeSeriesSplit` cross-validated hyperparameter tuning.  
> The improvement was remarkable:  
> - **Mean Absolute Error dropped by 40.8%**, from 0.518 kW down to 0.306 kW.  
> - **Root Mean Squared Error dropped by 30.5%**, from 0.638 kW down to 0.443 kW.  
> - **R-squared surged from 0.23 to 0.628**, meaning our engineered model explains nearly two-thirds of the total variance in electricity demand.  
> On the 14-day test forecast plot, you can clearly see the baseline linear model in red missing sharp peaks, while our tuned Random Forest in green closely traces the true actual consumption in black."

---

### [03:15 – 03:50] Scene 5: Feature Importance & Error Analysis
**Visual:** Display `09_feature_importance.png` and `outputs/reports/error_analysis.txt`.
> **Speaker:**  
> "Feature importance analysis revealed that `lag_1`—the immediate prior hour's load—is the single most influential predictor with 34% importance, followed closely by the 24-hour rolling mean.  
> Our error analysis showed that overnight hours between 1 AM and 6 AM have the lowest errors (MAE ~0.18 kW) due to predictable base loads, whereas evening peak hours between 6 PM and 10 PM experience higher variance due to sporadic household appliance usage."

---

### [03:50 – 04:15] Scene 6: Artifacts & Conclusion
**Visual:** Show `outputs/models/electricity_consumption_model.pkl` and GitHub-ready project structure.
> **Speaker:**  
> "The final tuned model is persisted as a serialized `.pkl` artifact, verified with test inference, and accompanied by full unit documentation, CSV reports, 9 high-res figures, and pre-rendered Jupyter notebooks.  
> This project successfully validates the core thesis: domain-driven feature engineering and thoughtful model selection are the single most impactful drivers of machine learning performance.  
> Thank you to AVIRENZA Technologies for this outstanding internship learning opportunity."
