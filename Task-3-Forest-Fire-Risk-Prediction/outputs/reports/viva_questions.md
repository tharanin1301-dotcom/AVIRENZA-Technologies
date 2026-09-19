# AVIRENZA Technologies – AI/ML Internship
## Task 3: Forest Fire Risk Prediction Using Machine Learning
### Viva Voce Preparation Guide (20 Questions & Answers)

**Intern:** Tharani Natarajan  
**College:** IFET College of Engineering  
**Department:** Artificial Intelligence and Data Science  
**Task:** Task 3 – Build a Supervised Learning Model (Intermediate)

---

### Q1: What is supervised learning?
**Answer:** Supervised learning is a subfield of machine learning where an algorithm learns a mapping function from input features ($X$) to known ground-truth target labels ($y$) using labeled training data. Once trained, the model predicts the target on previously unseen instances.

---

### Q2: What is classification in machine learning?
**Answer:** Classification is a supervised learning task where the target output consists of discrete categories or classes (e.g., binary: fire vs. no-fire, or multi-class) rather than continuous real values.

---

### Q3: What is the UCI Forest Fires dataset?
**Answer:** The UCI Forest Fires dataset contains 517 meteorological and environmental observations collected from Montesinho Natural Park in northeast Portugal (by Paulo Cortez and Aníbal de Morais, 2007). It records spatial coordinates, temporal indicators (month, day), Canadian Fire Weather Index (FWI) components (FFMC, DMC, DC, ISI), weather variables (temp, RH, wind, rain), and the total burned area.

---

### Q4: What is the target variable used in this project?
**Answer:** The target variable is `fire_occurrence`, a binary indicator derived from the observed burned `area`:
$$\text{fire\_occurrence} = \begin{cases} 1, & \text{if } \text{area} > 0 \text{ (Burned area recorded)} \\ 0, & \text{if } \text{area} = 0 \text{ (No burned area recorded)} \end{cases}$$

---

### Q5: Why was `fire_occurrence` created instead of directly predicting burned area?
**Answer:** The original `area` variable is heavily skewed with nearly 48% zero values (no fire) and extreme positive outliers (small fires vs. large burns). Transforming the problem into binary classification creates a well-defined intermediate supervised learning problem that models whether environmental conditions were associated with historical fire occurrence.

---

### Q6: Why is the original `area` column not used as a feature?
**Answer:** Using `area` as an input feature would introduce severe **target leakage** because `fire_occurrence` was directly derived from `area`. If `area` were in the feature matrix, the model would achieve artificial 100% accuracy by simply checking if $\text{area} > 0$, without learning any true meteorological relationships.

---

### Q7: What is target leakage and why is it dangerous?
**Answer:** Target leakage (or data leakage) occurs when information from the target variable or data that would not be available at inference time is inadvertently included in the training feature space. This results in unrealistically high training/validation scores that fail completely when deployed in real-world scenarios.

---

### Q8: What is Logistic Regression?
**Answer:** Logistic Regression is a linear classification algorithm that models the log-odds (logit) of the positive class as a linear combination of input features:
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k$$
It applies the sigmoid (logistic) function to output well-calibrated probabilities between 0 and 1.

---

### Q9: Why can Logistic Regression be used for classification even though it is called "regression"?
**Answer:** Despite its name, Logistic Regression performs classification by using linear regression on log-odds and mapping the resulting continuous value into a $[0, 1]$ probability using the standard logistic sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$, thresholding at $0.5$ (or custom threshold) to assign binary classes.

---

### Q10: What is Random Forest?
**Answer:** Random Forest is an ensemble learning method based on **bagging (Bootstrap Aggregation)**. It constructs a collection of decorrelated decision trees during training and outputs the majority vote (for classification) or average prediction (for regression) across all individual trees.

---

### Q11: Why use multiple decision trees instead of a single decision tree?
**Answer:** Individual decision trees have high variance and are prone to overfitting training data. Random Forest reduces variance without increasing bias by averaging across many diverse trees, each trained on bootstrap subsets of data and considering random feature subsets at each split.

---

### Q12: What is train-test splitting and why is it necessary?
**Answer:** Train-test splitting partitions the dataset into independent training (e.g., 80%) and testing (e.g., 20%) subsets. It simulates how well the model generalizes to previously unseen data and prevents over-optimistic evaluation on memorized training samples.

---

### Q13: Why did we use `stratify=y` during train-test splitting?
**Answer:** Stratified sampling ensures that the proportion of positive (fire) and negative (no fire) classes is identically maintained in both the training set and testing set, avoiding class distribution skew between train and test partitions.

---

### Q14: What is a Confusion Matrix?
**Answer:** A Confusion Matrix is a tabular layout summarizing classification performance by comparing actual target labels with model predictions across four quadrants:
1. **True Negatives (TN):** Actual No Fire correctly predicted as No Fire.
2. **False Positives (FP):** Actual No Fire incorrectly predicted as Fire (Type I Error).
3. **False Negatives (FN):** Actual Fire incorrectly predicted as No Fire (Type II Error).
4. **True Positives (TP):** Actual Fire correctly predicted as Fire.

---

### Q15: What is Precision and when is it important?
**Answer:** Precision measures the accuracy of positive predictions:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
It is critical when the cost of a false alarm (False Positive) is high, such as avoiding unnecessary deployment of emergency suppression resources.

---

### Q16: What is Recall (Sensitivity) and when is it critical?
**Answer:** Recall measures the proportion of actual positive instances correctly identified:
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
In safety-critical domains like fire risk detection, high recall is vital to minimize missed fires (False Negatives).

---

### Q17: What is the F1-Score?
**Answer:** The F1-Score is the harmonic mean of Precision and Recall:
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
It provides a single balanced metric, especially when precision and recall must be traded off against each other.

---

### Q18: What is ROC-AUC?
**Answer:** Receiver Operating Characteristic (ROC) curves plot the True Positive Rate (Recall) against the False Positive Rate across all possible probability thresholds. The Area Under the Curve (ROC-AUC) measures the model's aggregate ability to discriminate between positive and negative classes (1.0 = perfect, 0.5 = random guess).

---

### Q19: What is Feature Importance in Random Forest?
**Answer:** Feature importance (Mean Decrease in Impurity / Gini Importance) measures the total reduction in criterion impurity brought by all splits on a given feature, averaged across all trees in the ensemble. It ranks features by how heavily the forest relied on them for classification.

---

### Q20: Why does feature importance not prove causal relationships?
**Answer:** Feature importance reflects statistical associations and predictive utility within the specific dataset and model structure. It does not establish causal mechanisms because confounding variables, correlated features, or sample selection biases can cause non-causal variables to receive high importance scores.
