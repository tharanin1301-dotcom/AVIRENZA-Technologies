# Viva / Technical Interview Questions & Answers

**Task 4:** Electricity Consumption Prediction with Feature Engineering and Model Improvement  
**Internship:** AVIRENZA Technologies  
**Student:** Tharani Natarajan | IFET College of Engineering  
**Department:** Artificial Intelligence and Data Science  

---

### Q1: What is the primary purpose of Task 4?
**Answer:** The primary goal is to demonstrate the power of **Feature Engineering** and **Model Improvement** in machine learning. By starting with a naive baseline linear model using basic raw features ($R^2 \approx 0.23$), we show how formulating domain-specific temporal, cyclical, autocorrelation lag, and rolling window features—combined with non-linear tree ensembles and time-aware hyperparameter tuning—systematically improves predictive power ($R^2 \approx 0.63$, reducing MAE by ~41%).

---

### Q2: Why can we NOT use random `train_test_split` with shuffling in time-series data?
**Answer:** In time-series forecasting, observations exhibit temporal autocorrelation and causality. If we shuffle the dataset randomly, observations from future timestamps ($t+1, t+5$) would be placed in the training set while past timestamps ($t, t+2$) are in the test set. The model would learn from the future to predict the past, causing catastrophic **lookahead data leakage**. When deployed in the real world where future data is never accessible, the model's accuracy would collapse. We must always enforce **chronological splitting** (first 80% train, subsequent 20% test).

---

### Q3: How was data leakage prevented when creating rolling window statistics?
**Answer:** A common, subtle form of leakage occurs when computing moving averages or rolling statistics (e.g., 24-hour mean) that include the target value at time $t$ to predict time $t$. To prevent this, we strictly applied `target.shift(1)` before computing any rolling operations (`rolling_mean_24`, `rolling_std_24`, etc.). This guarantees that only historical values from $t-1$ and earlier are ever aggregated.

---

### Q4: Why are cyclical Sine and Cosine encodings used for periodic features like hour and month?
**Answer:** Traditional numeric encodings treat time linearly (e.g., hour 23 and hour 0 have a numerical difference of 23, even though they are adjacent consecutive hours). By transforming the hour into two dimensions using:
$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
the distance between 23:00 and 00:00 is smoothly preserved as a continuous circular function without artificial boundary discontinuities.

---

### Q5: What are lag features, and which lags did you choose?
**Answer:** Lag features capture autoregressive dependencies by feeding past values of the target variable directly into the feature matrix. We selected:
- `lag_1`, `lag_2`, `lag_3`: To capture immediate short-term hourly momentum.
- `lag_24`, `lag_48`: To capture daily 24-hour diurnal seasonality (what happened at the same hour yesterday and two days ago).
- `lag_168`: To capture weekly seasonality (what happened at the same hour on the exact same day last week).

---

### Q6: Why did you exclude Sub-metering features from the predictive feature matrix?
**Answer:** The dataset's sub-metering attributes (`Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`) measure active energy consumed by specific rooms (kitchen, laundry, AC/heating). Because total `Global_active_power` is essentially the sum of these sub-meterings plus general lighting, including concurrent sub-meterings would constitute **target leakage** (asking the model to predict the whole when it is already given the parts at the current timestamp). In real-world operational forecasting, sub-meter readings for the upcoming hour are unknown.

---

### Q7: Why did you use `TimeSeriesSplit` instead of standard $K$-Fold cross-validation?
**Answer:** Standard $K$-Fold CV randomly partitions the dataset into folds, which once again breaks temporal ordering and leaks future records into the training folds. `TimeSeriesSplit` uses an **expanding window** approach: fold 1 trains on slice $1$ and tests on slice $2$; fold 2 trains on slices $1+2$ and tests on slice $3$, and so forth. This strictly respects temporal ordering and simulates real-world forecasting.

---

### Q8: Why did Random Forest outperform Linear Regression so dramatically?
**Answer:** Electricity demand in households is fundamentally non-linear and discontinuous. When a heavy electrical appliance (e.g., an oven, electric stove, or water heater) is turned on, load jumps sharply by several kilowatts rather than increasing smoothly along a linear plane. Linear regression struggles to model these sharp step-functions and non-linear interactions between hour of day, day of week, and lag values. Random Forest, being an ensemble of decision trees, handles non-linear boundaries, threshold effects, and complex interactions naturally.

---

### Q9: What were the most important features according to the trained Random Forest model?
**Answer:** 
1. `lag_1` (Immediate preceding hour's consumption) was the single most dominant predictor (~34% relative importance), reflecting strong short-term inertia in household occupancy and heating.
2. `rolling_mean_24` (24-hour moving average), which establishes the prevailing baseline energy regime.
3. `lag_2` and `lag_3` (short-term momentum).
4. `rolling_max_24` (recent peak surge capacity).

---

### Q10: What did the error analysis reveal about model performance across different regimes?
**Answer:**
- **Off-Peak Hours (01:00–06:00):** Lowest MAE (~0.18 kW). Nighttime base load is mostly composed of steady background appliances (refrigerators, standby routers), making it highly predictable.
- **Evening Peak Hours (18:00–22:00):** Highest MAE (~0.44 kW). Evening hours involve unpredictable human behavior (spontaneous cooking, TV, laundry), creating stochastic spikes that have higher residual variance.

---

### Q11: How do you interpret MAE, RMSE, and $R^2$ in this context?
**Answer:**
- **MAE (Mean Absolute Error):** Represents the average magnitude of prediction error in actual kilowatts (kW). Our model achieved 0.3063 kW, meaning predictions are on average within ~306 Watts of true consumption.
- **RMSE (Root Mean Squared Error):** Penalizes larger outlier errors more heavily than MAE. Achieved 0.4433 kW.
- **$R^2$ (Coefficient of Determination):** Measures the proportion of variance in power consumption explained by the features. Baseline explained only 23% of variance ($R^2 = 0.2299$), while our feature-engineered model explains ~63% ($R^2 = 0.6284$).

---

### Q12: What hyperparameters were tuned in the Random Forest model?
**Answer:** Using `GridSearchCV` combined with a 3-split `TimeSeriesSplit`, we tuned:
- `n_estimators`: Tested 100 vs 150 (optimal was 150).
- `max_depth`: Tested 12 vs 16 (optimal was 16 to capture fine-grained interaction depth).
- `min_samples_split`: Tested 4 vs 8 (optimal was 4).

---

### Q13: Why did we downsample the dataset from 1-minute to 1-hour intervals?
**Answer:** The raw UCI dataset contains over 2 million 1-minute records. Minute-level electricity data contains extreme high-frequency white noise (e.g. compressors cycling on for 90 seconds). Aggregating to hourly averages smooths out sensor jitter, captures actionable demand-side planning horizons (which utilities actually use for grid scheduling), and reduces memory overhead while retaining 34,589 robust observations.

---

### Q14: How did you handle initial `NaN` values produced by lag and rolling operations?
**Answer:** Because our longest historical lag is `lag_168` (1 week / 168 hours) and rolling window is 168 hours, the first 168 rows of the dataset do not have complete historical feature data. We cleanly dropped these first 168 initial setup rows (`dropna()`), reducing the active dataset from 34,589 to 34,421 rows. Crucially, we aligned both the baseline dataset and the feature-engineered dataset to these exact same timestamps to guarantee an identical, mathematically fair test set evaluation.

---

### Q15: How would this model be deployed in a real-world smart grid or smart home system?
**Answer:** The serialized model artifact (`electricity_consumption_model.pkl`) would be integrated into an energy management service (e.g., via FastAPI or an edge gateway). At the top of every hour, the system queries the latest 168 hours of smart meter readings, computes the 26 engineered features, generates the next-hour forecast, and feeds this prediction to smart battery storage schedulers or demand-response thermostats to minimize peak grid electricity tariffs.
