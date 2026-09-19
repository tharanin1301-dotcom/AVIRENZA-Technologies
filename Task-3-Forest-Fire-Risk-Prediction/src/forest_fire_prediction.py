"""
AVIRENZA Technologies - AI/ML Internship
Task 3: Forest Fire Risk Prediction Using Machine Learning
Student: Tharani Natarajan
College: IFET College of Engineering
Department: Artificial Intelligence and Data Science

Description:
A modular, production-ready Supervised Machine Learning classification pipeline
predicting historical forest fire occurrence from environmental and meteorological
observations using the official UCI Forest Fires dataset.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Set style for professional figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load the UCI Forest Fires dataset from CSV.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    
    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset successfully loaded with shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive inspection of the raw dataset.
    """
    missing_total = int(df.isnull().sum().sum())
    exact_duplicates = int(df.duplicated().sum())

    inspection_results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "exact_duplicates": exact_duplicates,
        "numerical_stats": df.describe().to_dict(),
        "categorical_stats": df.describe(include=["object"]).to_dict() if df.select_dtypes(include=["object"]).shape[1] > 0 else {},
    }
    
    logger.info("--- DATASET INSPECTION ---")
    logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Missing values count across all columns: {missing_total}")
    logger.info(f"Exact duplicate rows detected: {exact_duplicates}")
    return inspection_results


def clean_data(df: pd.DataFrame, drop_exact_duplicates: bool = True) -> pd.DataFrame:
    """
    Perform necessary data validation and cleaning.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # Document duplicate check
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0 and drop_exact_duplicates:
        logger.info(f"Removing {dup_count} exact duplicate rows.")
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No duplicates removed.")
        
    logger.info(f"Cleaned dataset shape: {df_clean.shape} (from {initial_rows} initial rows)")
    return df_clean


def create_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create the binary classification target 'fire_occurrence'.
    area > 0  -> 1 (Fire / burned area recorded)
    area == 0 -> 0 (No burned area recorded)
    
    Strictly separates 'area' to prevent target leakage.
    """
    logger.info("Creating target variable 'fire_occurrence' from 'area' column.")
    y = (df["area"] > 0).astype(int)
    y.name = "fire_occurrence"
    
    # Drop 'area' from feature set to avoid target leakage
    X = df.drop(columns=["area"])
    
    class_counts = y.value_counts().to_dict()
    class_percentages = (y.value_counts(normalize=True) * 100).round(2).to_dict()
    
    logger.info(f"Target distribution -> Class 0 (No Fire): {class_counts.get(0, 0)} ({class_percentages.get(0, 0)}%), "
                f"Class 1 (Fire): {class_counts.get(1, 0)} ({class_percentages.get(1, 0)}%)")
    return X, y


def identify_features(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical feature groups.
    """
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    return numerical_features, categorical_features


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split to preserve class ratios.
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}, stratify=y")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def create_preprocessor(
    numerical_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build a leakage-free Scikit-learn ColumnTransformer preprocessor.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_logistic_regression_pipeline(
    preprocessor: ColumnTransformer, random_state: int = 42
) -> Pipeline:
    """
    Build Logistic Regression model pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state, max_iter=1000
                ),
            ),
        ]
    )
    return pipeline


def build_random_forest_pipeline(
    preprocessor: ColumnTransformer, random_state: int = 42, n_estimators: int = 100
) -> Pipeline:
    """
    Build Random Forest model pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Train model pipeline and evaluate comprehensive classification metrics on test set.
    """
    logger.info(f"Fitting {model_name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = (
        roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    )
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    logger.info(
        f"{model_name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}"
    )

    return {
        "model_name": model_name,
        "pipeline": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classification_report": clf_report,
        "confusion_matrix": conf_mat,
    }


def perform_cross_validation(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Perform 5-fold Stratified Cross-Validation on training data.
    """
    cv_stratified = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv_stratified, scoring="f1")
    roc_auc_scores = cross_val_score(model, X_train, y_train, cv=cv_stratified, scoring="roc_auc")
    acc_scores = cross_val_score(model, X_train, y_train, cv=cv_stratified, scoring="accuracy")

    return {
        "cv_f1_mean": float(np.mean(f1_scores)),
        "cv_f1_std": float(np.std(f1_scores)),
        "cv_roc_auc_mean": float(np.mean(roc_auc_scores)),
        "cv_roc_auc_std": float(np.std(roc_auc_scores)),
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_accuracy_std": float(np.std(acc_scores)),
    }


def plot_eda_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate professional EDA figures and save them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#2b5c8f", "#d95f02"]

    # 1. Fire Occurrence Target Distribution
    plt.figure(figsize=(7, 5), dpi=300)
    target = (df["area"] > 0).astype(int)
    counts = target.value_counts().sort_index()
    labels = ["No Fire (area = 0)", "Fire Occurred (area > 0)"]
    bars = plt.bar(labels, counts.values, color=palette, width=0.5, edgecolor="#333333", linewidth=1)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h} ({h/len(target)*100:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.title("Distribution of Historical Fire Occurrence Target", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Number of Observations", fontsize=11)
    plt.ylim(0, max(counts.values) + 40)
    plt.tight_layout()
    plt.savefig(output_dir / "01_fire_occurrence_distribution.png", dpi=300)
    plt.close()

    # 2. Temperature Distribution
    plt.figure(figsize=(7, 5), dpi=300)
    sns.histplot(df["temp"], kde=True, color="#d95f02", bins=25, edgecolor="#ffffff", alpha=0.7)
    plt.title("Ambient Temperature Distribution (°C)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Temperature (°C)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.axvline(df["temp"].mean(), color="#1f77b4", linestyle="--", linewidth=1.5, label=f"Mean: {df['temp'].mean():.1f}°C")
    plt.axvline(df["temp"].median(), color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Median: {df['temp'].median():.1f}°C")
    plt.legend(frameon=True, facecolor="#f9f9f9")
    plt.tight_layout()
    plt.savefig(output_dir / "02_temperature_distribution.png", dpi=300)
    plt.close()

    # 3. Relative Humidity Distribution
    plt.figure(figsize=(7, 5), dpi=300)
    sns.histplot(df["RH"], kde=True, color="#2b5c8f", bins=25, edgecolor="#ffffff", alpha=0.7)
    plt.title("Relative Humidity (RH %) Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Relative Humidity (%)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.axvline(df["RH"].mean(), color="#d95f02", linestyle="--", linewidth=1.5, label=f"Mean: {df['RH'].mean():.1f}%")
    plt.legend(frameon=True, facecolor="#f9f9f9")
    plt.tight_layout()
    plt.savefig(output_dir / "03_relative_humidity_distribution.png", dpi=300)
    plt.close()

    # 4. Fire Occurrence vs Temperature
    plt.figure(figsize=(7, 5), dpi=300)
    df_plot = df.copy()
    df_plot["fire_label"] = np.where(df_plot["area"] > 0, "Fire (area > 0)", "No Fire (area = 0)")
    sns.boxplot(x="fire_label", y="temp", data=df_plot, palette=palette, width=0.45, fliersize=4)
    plt.title("Ambient Temperature by Fire Occurrence Category", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Target Category", fontsize=11)
    plt.ylabel("Temperature (°C)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "04_fire_occurrence_vs_temperature.png", dpi=300)
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(9, 7), dpi=300)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap of Numerical Meteorological Variables", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / "05_correlation_heatmap.png", dpi=300)
    plt.close()


def plot_confusion_matrix(
    eval_result: Dict[str, Any], output_path: Path
) -> None:
    """
    Plot and save confusion matrix heatmap.
    """
    cm = eval_result["confusion_matrix"]
    model_name = eval_result["model_name"]
    
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted No Fire (0)", "Predicted Fire (1)"],
        yticklabels=["Actual No Fire (0)", "Actual Fire (1)"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curve(
    models_dict: Dict[str, Dict[str, Any]], y_test: pd.Series, output_paths: List[Path]
) -> None:
    """
    Plot ROC Curves comparing Logistic Regression and Random Forest.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#d95f02"}

    for name, res in models_dict.items():
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=colors.get(name, "#333333"),
                lw=2.2,
                label=f"{name} (AUC = {roc_auc_val:.3f})",
            )

    plt.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve Comparison", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()

    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()


def plot_feature_importance(
    rf_pipeline: Pipeline,
    numerical_features: List[str],
    categorical_features: List[str],
    output_paths: List[Path],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract and visualize feature importance from Random Forest using get_feature_names_out().
    """
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_clf = rf_pipeline.named_steps["classifier"]

    # Retrieve transformed feature names
    feature_names = preprocessor.get_feature_names_out()
    importances = rf_clf.feature_importances_

    # Clean up feature names for presentation (e.g. num__temp -> temp, cat__month_aug -> month_aug)
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    fi_df = pd.DataFrame({"Feature": clean_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(9, 6), dpi=300)
    top_fi = fi_df.head(top_n).sort_values(by="Importance", ascending=True)

    bars = plt.barh(top_fi["Feature"], top_fi["Importance"], color="#2b5c8f", edgecolor="#1a385c", height=0.6)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", ha="left", fontsize=9)

    plt.title(f"Random Forest Top {top_n} Feature Importances", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Gini Feature Importance Score", fontsize=11)
    plt.ylabel("Transformed Features", fontsize=11)
    plt.xlim(0, max(top_fi["Importance"]) * 1.15)
    plt.tight_layout()

    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()

    return fi_df


def generate_reports(
    results: Dict[str, Dict[str, Any]],
    cv_results: Dict[str, Dict[str, float]],
    class_dist: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Generate model_comparison.csv and model_evaluation.txt.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparison DataFrame
    comp_rows = []
    for name, res in results.items():
        cv = cv_results.get(name, {})
        comp_rows.append(
            {
                "Model": name,
                "Accuracy": round(res["accuracy"], 4),
                "Precision": round(res["precision"], 4),
                "Recall": round(res["recall"], 4),
                "F1-score": round(res["f1"], 4),
                "ROC-AUC": round(res["roc_auc"], 4),
                "CV_F1_Mean": round(cv.get("cv_f1_mean", 0.0), 4),
                "CV_F1_Std": round(cv.get("cv_f1_std", 0.0), 4),
                "CV_ROC_AUC_Mean": round(cv.get("cv_roc_auc_mean", 0.0), 4),
            }
        )
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(output_dir / "model_comparison.csv", index=False)
    logger.info(f"Saved model comparison table to: {output_dir / 'model_comparison.csv'}")

    # 2. Text Summary Report
    report_file = output_dir / "model_evaluation.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AVIRENZA TECHNOLOGIES AI/ML INTERNSHIP - TASK 3 REPORT\n")
        f.write("Project: Forest Fire Risk Prediction Using Machine Learning\n")
        f.write("Student: Tharani Natarajan | IFET College of Engineering\n")
        f.write("Department: Artificial Intelligence and Data Science\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DATASET & PROBLEM FORMULATION\n")
        f.write("-" * 40 + "\n")
        f.write("Dataset: Official UCI Forest Fires Dataset (Montesinho Natural Park, Portugal)\n")
        f.write(f"Class 0 (No Burned Area, area == 0): {class_dist['class_0_count']} ({class_dist['class_0_pct']}%)\n")
        f.write(f"Class 1 (Burned Area Recorded, area > 0): {class_dist['class_1_count']} ({class_dist['class_1_pct']}%)\n")
        f.write("Leakage Prevention: 'area' variable excluded strictly from feature matrix.\n\n")

        f.write("2. MODEL PERFORMANCE COMPARISON (TEST SET: 20% Stratified Split)\n")
        f.write("-" * 70 + "\n")
        f.write(comp_df.to_string(index=False))
        f.write("\n\n")

        for name, res in results.items():
            f.write(f"3. DETAILED CLASSIFICATION REPORT: {name.upper()}\n")
            f.write("-" * 50 + "\n")
            for target_class in ["0", "1"]:
                metrics = res["classification_report"][target_class]
                f.write(
                    f"Class {target_class} -> Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1-score']:.4f}, Support: {metrics['support']}\n"
                )
            macro = res["classification_report"]["macro avg"]
            f.write(f"Macro Avg -> Precision: {macro['precision']:.4f}, Recall: {macro['recall']:.4f}, F1-score: {macro['f1-score']:.4f}\n")
            weighted = res["classification_report"]["weighted avg"]
            f.write(f"Weighted Avg -> Precision: {weighted['precision']:.4f}, Recall: {weighted['recall']:.4f}, F1-score: {weighted['f1-score']:.4f}\n")
            f.write(f"Confusion Matrix (TN, FP, FN, TP):\n{res['confusion_matrix']}\n\n")

        f.write("4. 5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING SET\n")
        f.write("-" * 50 + "\n")
        for name, cv in cv_results.items():
            f.write(f"{name}:\n")
            f.write(f"  - F1-Score: {cv['cv_f1_mean']:.4f} (+/- {cv['cv_f1_std']:.4f})\n")
            f.write(f"  - ROC-AUC : {cv['cv_roc_auc_mean']:.4f} (+/- {cv['cv_roc_auc_std']:.4f})\n")
            f.write(f"  - Accuracy: {cv['cv_accuracy_mean']:.4f} (+/- {cv['cv_accuracy_std']:.4f})\n\n")

    logger.info(f"Saved evaluation text report to: {report_file}")


def save_model(pipeline: Pipeline, output_path: Path) -> None:
    """
    Persist the full Scikit-learn Pipeline (preprocessor + model) via joblib.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"Model pipeline successfully saved to: {output_path}")


def main() -> None:
    """
    Main orchestration function.
    """
    # Define directories relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "forestfires.csv"
    figures_dir = project_root / "outputs" / "figures"
    reports_dir = project_root / "outputs" / "reports"
    models_dir = project_root / "outputs" / "models"

    logger.info("=== AVIRENZA TECHNOLOGIES - TASK 3 FOREST FIRE PREDICTION ===")
    logger.info(f"Project root: {project_root}")

    # 1. Load Data
    df_raw = load_data(data_path)

    # 2. Inspect Data
    inspect_data(df_raw)

    # 3. Clean Data
    df = clean_data(df_raw, drop_exact_duplicates=True)

    # 4. Target & Features
    X, y = create_target(df)
    class_dist = {
        "class_0_count": int((y == 0).sum()),
        "class_0_pct": round((y == 0).mean() * 100, 2),
        "class_1_count": int((y == 1).sum()),
        "class_1_pct": round((y == 1).mean() * 100, 2),
    }

    num_features, cat_features = identify_features(X)

    # 5. EDA Figures
    logger.info("Generating Exploratory Data Analysis figures...")
    plot_eda_figures(df, figures_dir)

    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.20, random_state=42
    )

    # 7. Build Preprocessing & Pipelines
    preprocessor = create_preprocessor(num_features, cat_features)
    lr_pipeline = build_logistic_regression_pipeline(preprocessor, random_state=42)
    rf_pipeline = build_random_forest_pipeline(
        preprocessor, random_state=42, n_estimators=100
    )

    # 8. Train & Evaluate Models
    results = {}
    results["Logistic Regression"] = evaluate_model(
        lr_pipeline, X_train, y_train, X_test, y_test, "Logistic Regression"
    )
    results["Random Forest"] = evaluate_model(
        rf_pipeline, X_train, y_train, X_test, y_test, "Random Forest"
    )

    # 9. Cross-Validation
    cv_results = {}
    cv_results["Logistic Regression"] = perform_cross_validation(
        lr_pipeline, X_train, y_train, cv=5, random_state=42
    )
    cv_results["Random Forest"] = perform_cross_validation(
        rf_pipeline, X_train, y_train, cv=5, random_state=42
    )

    # 10. Generate Confusion Matrices
    plot_confusion_matrix(
        results["Logistic Regression"],
        figures_dir / "06_logistic_regression_confusion_matrix.png",
    )
    plot_confusion_matrix(
        results["Random Forest"],
        figures_dir / "07_random_forest_confusion_matrix.png",
    )

    # 11. Generate ROC Curve Comparison
    plot_roc_curve(
        results,
        y_test,
        [
            figures_dir / "08_roc_curve_comparison.png",
            figures_dir / "roc_curve_comparison.png",
        ],
    )

    # 12. Feature Importance for Random Forest
    fi_df = plot_feature_importance(
        rf_pipeline,
        num_features,
        cat_features,
        [
            figures_dir / "09_random_forest_feature_importance.png",
            figures_dir / "random_forest_feature_importance.png",
        ],
        top_n=15,
    )
    logger.info("Top 5 Transformed Features by Random Forest Importance:")
    for idx, row in fi_df.head(5).iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")

    # 13. Reports
    generate_reports(results, cv_results, class_dist, reports_dir)

    # 14. Save Model Pipeline
    model_output_path = models_dir / "random_forest_fire_model.pkl"
    save_model(rf_pipeline, model_output_path)

    # 15. Verify Model Loading
    loaded_pipeline = joblib.load(model_output_path)
    sample_preds = loaded_pipeline.predict(X_test.iloc[:5])
    logger.info(f"Model validation check: reload successful. Sample predictions: {sample_preds.tolist()}")

    logger.info("All Task 3 steps completed successfully.")


if __name__ == "__main__":
    main()
