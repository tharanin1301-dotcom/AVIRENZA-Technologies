# Bank Marketing Data Preprocessing - Viva Questions & Answers

**Student:** Tharani Natarajan  
**College:** IFET College of Engineering  
**Department:** Artificial Intelligence and Data Science  
**Internship:** AVIRENZA Technologies  
**Task:** Task 2 – Build a Data Preprocessing Pipeline  

---

### Q1: What is data preprocessing, and why is it essential in machine learning?
**Answer:** Data preprocessing is the technique of converting raw, unstructured, or noisy data into a clean, well-formatted, and numerical structure suitable for machine learning models. It is essential because real-world datasets often contain missing entries, inconsistent formats, unscaled numerical variables, and categorical text that algorithms cannot directly interpret.

---

### Q2: What is the UCI Bank Marketing dataset, and what is its business objective?
**Answer:** The Bank Marketing dataset is a real-world collection of direct phone-call marketing campaign records from a Portuguese banking institution (Moro et al., 2014). The business objective is to predict whether a client will subscribe to a long-term deposit (binary target `y`: `yes` or `no`).

---

### Q3: What is the target variable in this dataset, and how was it encoded?
**Answer:** The target variable is `y` with values `"yes"` (client subscribed) and `"no"` (client did not subscribe). It was preprocessed into binary numerical format: `no` $\rightarrow 0$, `yes` $\rightarrow 1$ using explicit mapping (`map({'no': 0, 'yes': 1})`).

---

### Q4: How many numerical and categorical features are present in the dataset?
**Answer:** The dataset contains:
- **7 Numerical Features:** `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, and `previous`.
- **9 Categorical Features:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, and `poutcome`.

---

### Q5: How did you handle "unknown" values in categorical attributes?
**Answer:** In the Bank Marketing dataset, `"unknown"` values (e.g., in `poutcome` at 81.75% and `contact` at 28.80%) are informative business categories representing real scenarios (such as clients never contacted before). Rather than dropping these rows or blindly replacing them with `NaN`, we treated `"unknown"` as a valid distinct category and encoded it via One-Hot Encoding so models retain this predictive signal.

---

### Q6: Why did you check for duplicate rows, and what was the outcome?
**Answer:** Checking for duplicate rows ensures data integrity and prevents artificial weighting or data leakage. The check (`df.duplicated().sum()`) confirmed 0 exact duplicate rows across all 45,211 records.

---

### Q7: Why is the dataset split into training and testing sets before applying transformations?
**Answer:** Splitting the data before fitting transformations is crucial to prevent **Data Leakage**. If scalers or imputers are fitted on the entire dataset, information about the test set's mean, variance, and distribution leaks into the training pipeline, leading to overly optimistic and flawed evaluation.

---

### Q8: What is stratified sampling (`stratify=y`), and why is it important here?
**Answer:** Stratified sampling ensures that both training and testing sets preserve the exact same class proportion as the original dataset. Since the target variable `y` is imbalanced (~88.3% `no` vs ~11.7% `yes`), stratification guarantees that both sets contain ~11.7% positive cases, avoiding sample bias.

---

### Q9: What imputation strategies were used for numerical and categorical features?
**Answer:** 
- **Numerical Features:** Imputed using `SimpleImputer(strategy="median")`, which is robust against extreme skewness and outliers in variables like `balance` and `duration`.
- **Categorical Features:** Imputed using `SimpleImputer(strategy="most_frequent")` (mode imputation).

---

### Q10: What is `StandardScaler`, and why is standardization applied to numerical features?
**Answer:** `StandardScaler` standardizes features by removing the mean and scaling to unit variance ($z = \frac{x - \mu}{\sigma}$). It prevents features with large numeric scales (e.g., `balance` with values up to 102,127) from dominating distance-based or gradient-based algorithms over smaller features like `campaign` or `age`.

---

### Q11: What is One-Hot Encoding, and how does `OneHotEncoder` handle unseen categories?
**Answer:** One-Hot Encoding converts categorical variables into binary dummy columns ($0$ or $1$). In our pipeline, `OneHotEncoder(handle_unknown="ignore")` was used so that if an unseen category appears during inference/testing, it creates all zeros rather than throwing a runtime error.

---

### Q12: What is Scikit-Learn's `ColumnTransformer`?
**Answer:** `ColumnTransformer` enables applying different transformation pipelines to distinct subsets of features (e.g., applying imputation + standard scaling to numerical columns while simultaneously applying imputation + one-hot encoding to categorical columns) within a single unified pipeline.

---

### Q13: What is Scikit-Learn's `Pipeline`?
**Answer:** A `Pipeline` chains together multiple data transformation steps and an estimator into a single, cohesive, sequential object. Calling `.fit()` on the pipeline sequentially fits all transformers and the final estimator, and `.transform()` applies them in order.

---

### Q14: How many features are produced after one-hot encoding and preprocessing?
**Answer:** The original 16 input features are expanded to **51 preprocessed features** due to the creation of one-hot indicator columns for all unique categorical levels across the 9 categorical features.

---

### Q15: What is an outlier, and why were extreme values in `balance` and `duration` not deleted?
**Answer:** An outlier is an observation that lies abnormally far from other values in the distribution. In banking data, high balances (e.g., €100,000+) and long call durations represent genuine, high-value customer interactions rather than data entry errors. Deleting them would discard critical business intelligence.

---

### Q16: How do you verify that no missing values exist after transformation?
**Answer:** By running `np.isnan(X_train_proc).sum()` and `np.isnan(X_test_proc).sum()`, both of which returned 0, verifying complete data integrity.

---

### Q17: Why was a small Logistic Regression model used during validation?
**Answer:** The baseline Logistic Regression model was executed strictly as a **preprocessing validation check** to verify that the transformed matrices are fully compatible with Scikit-learn estimators, yielding ~90.12% test accuracy and demonstrating clean ML-readiness.
