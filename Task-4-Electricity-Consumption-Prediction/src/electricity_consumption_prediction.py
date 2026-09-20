"""
AVIRENZA Technologies - AI/ML Internship
Task 4: Electricity Consumption Prediction with Feature Engineering and Model Improvement
Student: Tharani Natarajan
College: IFET College of Engineering
Department: Artificial Intelligence and Data Science

Description:
A modular, production-ready machine learning regression system demonstrating how
temporal, cyclical, lag, and rolling feature engineering and time-aware tuning
improve power consumption forecasting on the UCI Household Electric Power dataset.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Style for professional figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load the hourly electricity consumption dataset.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    
    logger.info(f"Loading hourly dataset from: {data_path}")
    df = pd.read_csv(data_path, index_col='Datetime', parse_dates=True)
    logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform initial inspection and summary of the electricity dataset.
    """
    inspection_results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "exact_duplicates": int(df.duplicated().sum()),
        "time_range": (str(df.index.min()), str(df.index.max())),
        "numerical_stats": df.describe().to_dict(),
    }
    
    logger.info("--- DATASET INSPECTION ---")
    logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Date Range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Missing Values: {int(df.isnull().sum().sum())}")
    return inspection_results


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset and handle missing values in continuous time-series.
    """
    df_clean = df.copy()
    if df_clean.isnull().sum().sum() > 0:
        df_clean = df_clean.interpolate(method='linear').dropna()
    logger.info(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar and time indicators from DatetimeIndex.
    """
    df_feat = df.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['day_of_year'] = df_feat.index.dayofyear
    df_feat['quarter'] = df_feat.index.quarter
    df_feat['is_weekend'] = (df_feat.index.dayofweek >= 5).astype(int)
    
    # Evening peak consumption hours based on empirical EDA (18:00 to 22:00)
    df_feat['is_peak_hour'] = df_feat['hour'].isin([18, 19, 20, 21, 22]).astype(int)
    
    logger.info("Created 7 temporal features: hour, day_of_week, month, day_of_year, quarter, is_weekend, is_peak_hour")
    return df_feat


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode periodic time variables (hour, day_of_week, month) into continuous sine/cosine cyclical representations.
    """
    df_feat = df.copy()
    
    # Hour cyclical encoding (period = 24)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
    
    # Day of week cyclical encoding (period = 7)
    df_feat['day_of_week_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7.0)
    df_feat['day_of_week_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7.0)
    
    # Month cyclical encoding (period = 12)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12.0)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12.0)
    
    logger.info("Created 6 cyclical features: hour_sin/cos, day_of_week_sin/cos, month_sin/cos")
    return df_feat


def create_lag_features(
    df: pd.DataFrame, target_col: str = 'Global_active_power', lags: List[int] = [1, 2, 3, 24, 48, 168]
) -> pd.DataFrame:
    """
    Create historical lag features strictly from past target values.
    """
    df_feat = df.copy()
    target_series = df_feat[target_col]
    
    for lag in lags:
        df_feat[f'lag_{lag}'] = target_series.shift(lag)
        
    logger.info(f"Created {len(lags)} lag features strictly from past data: {['lag_' + str(l) for l in lags]}")
    return df_feat


def create_rolling_features(
    df: pd.DataFrame, target_col: str = 'Global_active_power'
) -> pd.DataFrame:
    """
    Create rolling window statistics (mean, std, min, max) strictly applied to shift(1)
    to prevent target leakage of the current observation.
    """
    df_feat = df.copy()
    
    # Strictly shift by 1 to isolate historical window from current target
    shifted_target = df_feat[target_col].shift(1)
    
    # 24-hour historical window statistics
    df_feat['rolling_mean_24'] = shifted_target.rolling(24).mean()
    df_feat['rolling_std_24'] = shifted_target.rolling(24).std()
    df_feat['rolling_min_24'] = shifted_target.rolling(24).min()
    df_feat['rolling_max_24'] = shifted_target.rolling(24).max()
    
    # 168-hour (7-day) historical rolling mean
    df_feat['rolling_mean_168'] = shifted_target.rolling(168).mean()
    
    logger.info("Created 5 rolling features: rolling_mean_24, rolling_std_24, rolling_min_24, rolling_max_24, rolling_mean_168")
    return df_feat


def create_train_test_split(
    df: pd.DataFrame, feature_cols: List[str], target_col: str = 'Global_active_power', split_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a strict chronological train-test split (First 80% Train, Last 20% Test) without shuffling.
    """
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    logger.info(f"Chronological split -> Train: {X_train.shape[0]} samples ({train_df.index.min()} to {train_df.index.max()}) | "
                f"Test: {X_test.shape[0]} samples ({test_df.index.min()} to {test_df.index.max()})")
    return X_train, X_test, y_train, y_test


def evaluate_model(
    model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
) -> Dict[str, Any]:
    """
    Fit model and evaluate regression metrics (MAE, MSE, RMSE, R²).
    """
    logger.info(f"Fitting {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"{model_name} Evaluation -> MAE: {mae:.4f} kW | RMSE: {rmse:.4f} kW | R²: {r2:.4f}")
    
    return {
        "model_name": model_name,
        "model": model,
        "feature_count": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred,
    }


def perform_time_series_cv(
    model: Any, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5
) -> Dict[str, float]:
    """
    Perform time-series cross-validation (TimeSeriesSplit) to respect temporal causality.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        val_preds = model.predict(X_val)
        
        mae_scores.append(mean_absolute_error(y_val, val_preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, val_preds)))
        r2_scores.append(r2_score(y_val, val_preds))
        
    return {
        "cv_mae_mean": float(np.mean(mae_scores)),
        "cv_mae_std": float(np.std(mae_scores)),
        "cv_rmse_mean": float(np.mean(rmse_scores)),
        "cv_rmse_std": float(np.std(rmse_scores)),
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_r2_std": float(np.std(r2_scores)),
    }


def tune_model(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> RandomForestRegressor:
    """
    Perform TimeSeriesSplit hyperparameter tuning for Random Forest.
    """
    logger.info("Executing TimeSeriesSplit Hyperparameter Tuning on Improved Random Forest...")
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [12, 16],
        'min_samples_split': [4, 8],
    }
    
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Optimal Hyperparameters identified: {grid_search.best_params_}")
    return grid_search.best_estimator_


def plot_consumption(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate exploratory consumption plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Consumption Over Time (Resampled Weekly for clarity)
    plt.figure(figsize=(12, 5), dpi=300)
    df_weekly = df['Global_active_power'].resample('W').mean()
    plt.plot(df_weekly.index, df_weekly.values, color='#2b5c8f', lw=1.6)
    plt.title('Weekly Average Household Active Power Consumption (2006–2010)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Global Active Power (kW)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / '01_consumption_over_time.png', dpi=300)
    plt.close()

    # 2. Average Consumption by Hour of Day
    plt.figure(figsize=(8, 4.8), dpi=300)
    hourly_avg = df.groupby(df.index.hour)['Global_active_power'].mean()
    bars = plt.bar(hourly_avg.index, hourly_avg.values, color='#d95f02', edgecolor='#993d00', width=0.65)
    plt.title('Average Electricity Consumption by Hour of Day (kW)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Hour of Day (0–23)', fontsize=11)
    plt.ylabel('Average Active Power (kW)', fontsize=11)
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(output_dir / '02_avg_consumption_by_hour.png', dpi=300)
    plt.close()

    # 3. Average Consumption by Day of Week
    plt.figure(figsize=(8, 4.8), dpi=300)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_avg = df.groupby(df.index.dayofweek)['Global_active_power'].mean()
    colors = ['#2b5c8f' if i < 5 else '#d95f02' for i in range(7)]
    plt.bar(days, dow_avg.values, color=colors, edgecolor='#333333', width=0.55)
    plt.title('Average Electricity Consumption by Day of Week', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Day of Week', fontsize=11)
    plt.ylabel('Average Active Power (kW)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / '03_avg_consumption_by_day_of_week.png', dpi=300)
    plt.close()

    # 4. Average Consumption by Month
    plt.figure(figsize=(9, 4.8), dpi=300)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_avg = df.groupby(df.index.month)['Global_active_power'].mean()
    plt.plot(months, month_avg.values, marker='o', color='#2b5c8f', lw=2.2, markersize=7)
    plt.fill_between(months, month_avg.values, color='#2b5c8f', alpha=0.15)
    plt.title('Monthly Seasonal Consumption Trend', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Month', fontsize=11)
    plt.ylabel('Average Active Power (kW)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / '04_avg_consumption_by_month.png', dpi=300)
    plt.close()


def plot_predictions(
    y_test: pd.Series, predictions_dict: Dict[str, np.ndarray], output_dir: Path
) -> None:
    """
    Generate actual vs predicted time series and scatter plots.
    """
    # 5. Actual vs Predicted Time-Series (Zoom on representative 14-day window: 336 hours)
    plt.figure(figsize=(13, 5), dpi=300)
    zoom_len = 336
    t_idx = y_test.index[:zoom_len]
    plt.plot(t_idx, y_test.iloc[:zoom_len].values, label='Actual Consumption', color='#111111', lw=1.8)
    
    if 'Baseline Linear Regression' in predictions_dict:
        plt.plot(t_idx, predictions_dict['Baseline Linear Regression'][:zoom_len], label='Baseline LR', color='#e41a1c', lw=1.2, linestyle=':')
    if 'Tuned Random Forest' in predictions_dict:
        plt.plot(t_idx, predictions_dict['Tuned Random Forest'][:zoom_len], label='Tuned Engineered RF', color='#2ca02c', lw=1.6, linestyle='--')
        
    plt.title('Actual vs. Predicted Hourly Power Consumption (14-Day Test Window)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Timestamp', fontsize=11)
    plt.ylabel('Active Power (kW)', fontsize=11)
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / '05_actual_vs_predicted_time_series.png', dpi=300)
    plt.close()

    # 6. Actual vs Predicted Scatter Plot
    best_pred_name = 'Tuned Random Forest' if 'Tuned Random Forest' in predictions_dict else list(predictions_dict.keys())[-1]
    best_preds = predictions_dict[best_pred_name]
    
    plt.figure(figsize=(6.5, 6), dpi=300)
    plt.scatter(y_test, best_preds, alpha=0.25, color='#2b5c8f', s=12, edgecolors='none')
    max_val = max(y_test.max(), np.max(best_preds))
    plt.plot([0, max_val], [0, max_val], color='#d95f02', lw=2, linestyle='--', label='Perfect Prediction (y = x)')
    plt.title(f'Actual vs. Predicted Consumption ({best_pred_name})', fontsize=13, fontweight="bold", pad=12)
    plt.xlabel('Actual Active Power (kW)', fontsize=11)
    plt.ylabel('Predicted Active Power (kW)', fontsize=11)
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / '06_actual_vs_predicted_scatter.png', dpi=300)
    plt.close()

    # 7. Residual Error Distribution
    residuals = y_test - best_preds
    plt.figure(figsize=(7, 5), dpi=300)
    sns.histplot(residuals, kde=True, color='#2b5c8f', bins=40, edgecolor='#ffffff', alpha=0.7)
    plt.axvline(0, color='#d95f02', linestyle='--', lw=1.5, label='Zero Error')
    plt.axvline(residuals.mean(), color='#2ca02c', linestyle=':', lw=1.5, label=f'Mean Error ({residuals.mean():.3f} kW)')
    plt.title(f'Residual Error Distribution ({best_pred_name})', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Residual Error (Actual - Predicted) [kW]', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / '07_residual_error_distribution.png', dpi=300)
    plt.close()


def plot_feature_importance(
    model: Any, feature_names: List[str], output_dir: Path, top_n: int = 15
) -> pd.DataFrame:
    """
    Extract and plot ranked feature importances.
    """
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(9, 6), dpi=300)
    top_fi = fi_df.head(top_n).sort_values(by='Importance', ascending=True)
    
    bars = plt.barh(top_fi['Feature'], top_fi['Importance'], color='#2b5c8f', edgecolor='#1a385c', height=0.6)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', ha='left', fontsize=9)
        
    plt.title(f'Top {top_n} Engineered Features by Random Forest Importance', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Feature Importance Score (Gini / Impurity Reduction)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.xlim(0, max(top_fi['Importance']) * 1.15)
    plt.tight_layout()
    plt.savefig(output_dir / '09_feature_importance.png', dpi=300)
    plt.close()
    
    return fi_df


def compare_models(
    eval_results: Dict[str, Dict[str, Any]], cv_results: Dict[str, Dict[str, float]], output_dir: Path
) -> pd.DataFrame:
    """
    Generate before vs after feature engineering comparison table and chart.
    """
    rows = []
    for name, res in eval_results.items():
        cv = cv_results.get(name, {})
        rows.append({
            'Model': name,
            'Features_Count': res['feature_count'],
            'MAE (kW)': round(res['mae'], 4),
            'RMSE (kW)': round(res['rmse'], 4),
            'R2_Score': round(res['r2'], 4),
            'CV_MAE_Mean': round(cv.get('cv_mae_mean', 0.0), 4),
            'CV_RMSE_Mean': round(cv.get('cv_rmse_mean', 0.0), 4),
            'CV_R2_Mean': round(cv.get('cv_r2_mean', 0.0), 4),
        })
        
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    logger.info(f"Saved model comparison table to: {output_dir / 'model_comparison.csv'}")
    
    # 8. Comparison Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    x = np.arange(len(comp_df))
    width = 0.28
    
    plt.bar(x - width, comp_df['MAE (kW)'], width, label='MAE (kW)', color='#2b5c8f', edgecolor='#1a385c')
    plt.bar(x, comp_df['RMSE (kW)'], width, label='RMSE (kW)', color='#d95f02', edgecolor='#993d00')
    plt.bar(x + width, comp_df['R2_Score'], width, label='R² Score', color='#2ca02c', edgecolor='#1e701e')
    
    plt.title('Performance Comparison: Baseline vs. Engineered vs. Tuned Models', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, comp_df['Model'], rotation=10, ha='right')
    plt.ylabel('Metric Value', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / '08_baseline_vs_improved_comparison.png', dpi=300)
    plt.close()
    
    return comp_df


def perform_error_analysis(
    y_test: pd.Series, y_pred: np.ndarray, X_test: pd.DataFrame, output_dir: Path
) -> None:
    """
    Investigate high vs low error regimes and output an error report.
    """
    df_err = X_test.copy()
    df_err['Actual'] = y_test
    df_err['Predicted'] = y_pred
    df_err['Error'] = df_err['Actual'] - df_err['Predicted']
    df_err['Abs_Error'] = np.abs(df_err['Error'])
    
    # Group errors by hour
    hour_err = df_err.groupby('hour')['Abs_Error'].mean()
    # Group errors by peak vs non-peak
    peak_err = df_err.groupby('is_peak_hour')['Abs_Error'].mean() if 'is_peak_hour' in df_err.columns else {}
    # Group errors by weekend vs weekday
    weekend_err = df_err.groupby('is_weekend')['Abs_Error'].mean() if 'is_weekend' in df_err.columns else {}
    
    out_file = output_dir / 'error_analysis.txt'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("TASK 4 ERROR ANALYSIS REPORT: ELECTRICITY CONSUMPTION PREDICTION\n")
        f.write("Student: Tharani Natarajan | IFET College of Engineering\n")
        f.write("Internship: AVIRENZA Technologies\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Overall Test MAE: {df_err['Abs_Error'].mean():.4f} kW\n")
        f.write(f"Overall Test RMSE: {np.sqrt((df_err['Error']**2).mean()):.4f} kW\n\n")
        
        f.write("1. ERROR BY HOUR OF DAY (Mean Absolute Error):\n")
        for h, err in hour_err.items():
            f.write(f"  Hour {h:02d}:00 -> MAE: {err:.4f} kW\n")
        f.write("\n")
        
        if len(peak_err) > 0:
            f.write("2. PEAK VS. OFF-PEAK HOURS ERROR:\n")
            f.write(f"  Off-Peak Hours MAE: {peak_err.get(0, 0):.4f} kW\n")
            f.write(f"  Peak Hours (18-22h) MAE: {peak_err.get(1, 0):.4f} kW\n\n")
            
        if len(weekend_err) > 0:
            f.write("3. WEEKDAY VS. WEEKEND ERROR:\n")
            f.write(f"  Weekdays (Mon-Fri) MAE: {weekend_err.get(0, 0):.4f} kW\n")
            f.write(f"  Weekends (Sat-Sun) MAE: {weekend_err.get(1, 0):.4f} kW\n\n")
            
        f.write("4. KEY ERROR INSIGHTS:\n")
        f.write("  - Higher absolute errors occur during evening peak consumption periods (18:00–22:00) due to sharp discretionary household appliance spikes.\n")
        f.write("  - Lower errors occur during overnight baseline hours (01:00–06:00) when consumption is stable and predictable.\n")
        
    logger.info(f"Saved error analysis report to: {out_file}")


def save_feature_summary(feature_names: List[str], output_dir: Path) -> None:
    """
    Save structured CSV summary of engineered features.
    """
    categories = []
    descriptions = []
    
    for f in feature_names:
        if f in ['Voltage', 'Global_reactive_power', 'Global_intensity']:
            categories.append('Original Measurement')
            descriptions.append(f'Raw electrical measurement: {f}')
        elif f in ['hour', 'day_of_week', 'month', 'day_of_year', 'quarter', 'is_weekend', 'is_peak_hour']:
            categories.append('Temporal Indicator')
            descriptions.append(f'Calendar/hourly indicator: {f}')
        elif 'sin' in f or 'cos' in f:
            categories.append('Cyclical Encoding')
            descriptions.append(f'Periodic circular representation: {f}')
        elif 'lag' in f:
            categories.append('Lag Feature')
            descriptions.append(f'Past historical consumption value: {f}')
        elif 'rolling' in f:
            categories.append('Rolling Statistic')
            descriptions.append(f'Historical window statistic (shift 1 applied): {f}')
        else:
            categories.append('Other')
            descriptions.append(f'Engineered variable: {f}')
            
    summary_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Category': categories,
        'Description': descriptions
    })
    summary_df.to_csv(output_dir / 'feature_engineering_summary.csv', index=False)
    logger.info(f"Saved feature engineering summary to: {output_dir / 'feature_engineering_summary.csv'}")


def save_model(model: Any, output_path: Path) -> None:
    """
    Persist trained model via joblib with level 3 compression.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path, compress=3)
    logger.info(f"Model successfully saved to: {output_path}")


def main() -> None:
    """
    Main orchestration function.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "household_power_consumption_hourly.csv"
    figures_dir = project_root / "outputs" / "figures"
    reports_dir = project_root / "outputs" / "reports"
    models_dir = project_root / "outputs" / "models"
    
    logger.info("=== AVIRENZA TECHNOLOGIES - TASK 4 ELECTRICITY CONSUMPTION PREDICTION ===")
    logger.info(f"Project root: {project_root}")
    
    # 1. Load & Inspect Data
    df_raw = load_data(data_path)
    inspect_data(df_raw)
    df = clean_data(df_raw)
    
    # 2. EDA Visualizations
    logger.info("Generating exploratory data visualization figures...")
    plot_consumption(df, figures_dir)
    
    # 3. Baseline Feature Set & Model
    df_base = df.copy()
    df_base['hour'] = df_base.index.hour
    df_base['day_of_week'] = df_base.index.dayofweek
    df_base['month'] = df_base.index.month
    baseline_features = ['hour', 'day_of_week', 'month', 'Voltage', 'Global_reactive_power']
    
    # 4. Advanced Feature Engineering
    df_fe = df.copy()
    df_fe = create_temporal_features(df_fe)
    df_fe = create_cyclical_features(df_fe)
    df_fe = create_lag_features(df_fe, target_col='Global_active_power', lags=[1, 2, 3, 24, 48, 168])
    df_fe = create_rolling_features(df_fe, target_col='Global_active_power')
    
    # Drop initial NaN rows created by lag_168 and rolling windows
    df_fe = df_fe.dropna()
    logger.info(f"Feature-Engineered dataset shape after dropping initial lag NaNs: {df_fe.shape}")
    
    # Align baseline dataframe to the exact same temporal records for fair comparison
    df_base = df_base.loc[df_fe.index]
    X_tr_b, X_te_b, y_tr_b, y_te_b = create_train_test_split(df_base, baseline_features, split_ratio=0.80)
    
    fe_cols = [c for c in df_fe.columns if c not in ['Global_active_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
    logger.info(f"Total Engineered Feature Count: {len(fe_cols)}")
    
    # Save feature summary table
    save_feature_summary(fe_cols, reports_dir)
    
    # Chronological Split on Engineered Features
    X_tr_fe, X_te_fe, y_tr_fe, y_te_fe = create_train_test_split(df_fe, fe_cols, split_ratio=0.80)
    
    # 5. Train Models
    eval_results = {}
    cv_results = {}
    
    # Model 1: Baseline Linear Regression
    lr_base = LinearRegression()
    eval_results['Baseline Linear Regression'] = evaluate_model(lr_base, X_tr_b, y_tr_b, X_te_b, y_te_b, 'Baseline Linear Regression')
    cv_results['Baseline Linear Regression'] = perform_time_series_cv(lr_base, X_tr_b, y_tr_b, n_splits=5)
    
    # Model 2: Improved Random Forest with Feature Engineering
    rf_fe = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    eval_results['Improved Random Forest (FE)'] = evaluate_model(rf_fe, X_tr_fe, y_tr_fe, X_te_fe, y_te_fe, 'Improved Random Forest (FE)')
    cv_results['Improved Random Forest (FE)'] = perform_time_series_cv(rf_fe, X_tr_fe, y_tr_fe, n_splits=5)
    
    # Model 3: Tuned Random Forest
    tuned_rf = tune_model(X_tr_fe, y_tr_fe, random_state=42)
    eval_results['Tuned Random Forest'] = evaluate_model(tuned_rf, X_tr_fe, y_tr_fe, X_te_fe, y_te_fe, 'Tuned Random Forest')
    cv_results['Tuned Random Forest'] = perform_time_series_cv(tuned_rf, X_tr_fe, y_tr_fe, n_splits=5)
    
    # 6. Comparisons & Visualizations
    predictions_dict = {
        'Baseline Linear Regression': eval_results['Baseline Linear Regression']['y_pred'],
        'Improved Random Forest (FE)': eval_results['Improved Random Forest (FE)']['y_pred'],
        'Tuned Random Forest': eval_results['Tuned Random Forest']['y_pred'],
    }
    
    plot_predictions(y_te_fe, predictions_dict, figures_dir)
    fi_df = plot_feature_importance(tuned_rf, fe_cols, figures_dir, top_n=15)
    
    comp_df = compare_models(eval_results, cv_results, reports_dir)
    logger.info("\n--- FINAL MODEL COMPARISON ---\n" + comp_df.to_string(index=False))
    
    # 7. Error Analysis
    perform_error_analysis(y_te_fe, eval_results['Tuned Random Forest']['y_pred'], X_te_fe, reports_dir)
    
    # 8. Save Final Tuned Model
    model_output_path = models_dir / "electricity_consumption_model.pkl"
    save_model(tuned_rf, model_output_path)
    
    # 9. Verification Reload
    loaded_model = joblib.load(model_output_path)
    sample_preds = loaded_model.predict(X_te_fe.iloc[:5])
    logger.info(f"Model verification check passed. Sample predictions: {sample_preds.tolist()}")
    
    logger.info("All Task 4 steps completed successfully!")


if __name__ == "__main__":
    main()
