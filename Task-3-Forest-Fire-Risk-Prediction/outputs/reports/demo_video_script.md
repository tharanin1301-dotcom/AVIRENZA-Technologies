# Task 3: Forest Fire Risk Prediction Using Machine Learning
## Video Presentation & Demo Recording Script (2–3 Minutes)

**Intern:** Tharani Natarajan  
**College:** IFET College of Engineering  
**Department:** Artificial Intelligence and Data Science  
**Internship:** AVIRENZA Technologies (AI/ML Track)  
**Task:** Task 3 – Build a Supervised Learning Model (Intermediate)

---

## 🎬 Recording Overview & Timeline

| Segment | Duration | Screen to Show | Main Focus |
| :--- | :---: | :--- | :--- |
| **1. Introduction** | 0:00 - 0:25 | GitHub Repo / Root `README.md` | Name, College, Internship, Project Title |
| **2. Problem & Dataset** | 0:25 - 0:50 | Jupyter Notebook (Sections 1–4) | UCI Dataset, Target definition, Leakage prevention |
| **3. ML Pipeline Architecture** | 0:50 - 1:20 | Code in Notebook / `src/forest_fire_prediction.py` | ColumnTransformer, Imputation, Scaling, One-Hot Encoding |
| **4. Model Evaluation & Comparison** | 1:20 - 1:55 | Metrics Table, ROC Curve, Confusion Matrix | Logistic Regression vs. Random Forest (0.7037 AUC) |
| **5. Feature Importance & Findings** | 1:55 - 2:25 | Feature Importance Plot | Temp, RH, DC impact & Gini scores |
| **6. Conclusion & Wrap-up** | 2:25 - 2:45 | `outputs/models/` & Serialized `.pkl` file | End-to-end pipeline, key takeaways |

---

## 🎙️ Spoken Script (Word-for-Word Guide)

### 1. Introduction (0:00 - 0:25)
> *"Hello everyone! My name is **Tharani Natarajan**, from **IFET College of Engineering**, Department of Artificial Intelligence and Data Science.  
> As part of my **AI/ML Internship at AVIRENZA Technologies**, I am presenting **Task 3: Forest Fire Risk Prediction Using Machine Learning**."*

---

### 2. Problem Formulation & Leakage Prevention (0:25 - 0:50)
> *"In this project, we utilize the official **UCI Forest Fires Dataset** from Montesinho Natural Park in Portugal.  
> The original dataset records meteorological features and the burned area in hectares. For this supervised classification study, we formulated a binary target called `fire_occurrence` — where `1` indicates burned area was recorded and `0` indicates no fire.  
> **Crucially, to eliminate target leakage**, the original `area` variable is strictly excluded from our feature set so our model learns genuine meteorological relationships."*

---

### 3. Preprocessing & Leakage-Free Pipeline (0:50 - 1:20)
> *(Scroll to Section 13/14 in Notebook)*  
> *"We split our data into an 80% training set and a 20% stratified test set.  
> Using Scikit-Learn's `ColumnTransformer`, we constructed a clean, reproducible preprocessing pipeline:
> - **Numerical features** undergo median imputation and standard scaling.
> - **Categorical features** like month and day undergo mode imputation and one-hot encoding.  
> All transformations are fit strictly on the training partition and evaluated across two models: **Logistic Regression** and **Random Forest Classifier**."*

---

### 4. Model Evaluation & Comparison (1:20 - 1:55)
> *(Show the Model Comparison Table and ROC Curve)*  
> *"Evaluating on the independent test set:
> - **Logistic Regression** achieved an accuracy of **54.37%** and an ROC-AUC of **0.6096**.
> - **Random Forest** achieved an accuracy of **64.08%** and an ROC-AUC of **0.7037**.  
> In addition, 5-fold Stratified Cross-Validation confirmed consistent generalization performance without overfitting."*

---

### 5. Feature Importance Analysis (1:55 - 2:25)
> *(Show `random_forest_feature_importance.png`)*  
> *"Using `get_feature_names_out()`, we extracted the Gini feature importance from the Random Forest model:
> - **Ambient Temperature (`temp`)** was the most influential factor with an importance score of **0.1314**,
> - Followed by **Relative Humidity (`RH`)** at **0.1234**,
> - And drought indices including the **Drought Code (`DC`)** and **Duff Moisture Code (`DMC`)**.  
> This clearly reflects that high heat combined with dry fuel conditions strongly correlates with historical fire occurrences."*

---

### 6. Conclusion & Model Export (2:25 - 2:45)
> *(Show `outputs/models/random_forest_fire_model.pkl` in folder / VS Code)*  
> *"Finally, we serialized the complete end-to-end pipeline using `joblib` into `random_forest_fire_model.pkl` for reproducible inference.  
> All source code, 11 high-resolution figures, evaluation reports, and a complete 25-section Jupyter notebook are available in my GitHub repository.  
> Thank you to **AVIRENZA Technologies** for this practical learning opportunity!"*

---

## 💡 Practical Recording Tips

1. **Resolution:** Record in 1080p (1920x1080) with full screen or side-by-side IDE + notebook.
2. **Audio:** Use a clean microphone or headset and speak at a steady, confident pace.
3. **Cursor:** Use your mouse pointer to highlight key numbers (Accuracy: 64.08%, ROC-AUC: 0.7037, Top features).
4. **Tools to Record:** You can use **OBS Studio**, **Xbox Game Bar (Win + G)**, or **Clipchamp**.
