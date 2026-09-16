# World Happiness Analysis Using Python

<div align="center">

**Task 1 – Exploratory Data Analysis on a Real Dataset**  
**AVIRENZA Technologies | AI/ML Internship**

</div>

---

## Project Overview

This project performs a complete **Exploratory Data Analysis (EDA)** on the **World Happiness Report 2023** dataset.
It investigates the socioeconomic and social factors associated with national happiness scores across 102 countries.

---

## Internship Details

| Field | Details |
|-------|---------|
| **Company** | AVIRENZA Technologies |
| **Program** | AI/ML Internship |
| **Task** | Task 1 – EDA on a Real Dataset |
| **Author** | Tharani Natarajan |
| **Institution** | IFET College of Engineering |
| **Department** | Artificial Intelligence and Data Science |

---

## Problem Statement

Different countries show widely varying levels of happiness. Understanding which socioeconomic
factors (GDP, social support, freedom, health) are strongly associated with happiness can
help researchers and policymakers identify areas worth investigating further.

---

## Objectives

1. Load and inspect the World Happiness dataset.
2. Perform thorough data cleaning and preprocessing.
3. Compute descriptive statistics for all variables.
4. Create at least four professional visualizations.
5. Analyze correlations between happiness and socioeconomic indicators.
6. Extract clear, evidence-based findings.

---

## Dataset

| Field | Details |
|-------|---------|
| **Name** | World Happiness Report 2023 |
| **Source** | https://worldhappiness.report/data/ |
| **Year** | 2023 (Gallup World Poll) |
| **Countries** | 102 |
| **File** | `data/world_happiness_2023.csv` |

### Columns

| Column | Description |
|--------|-------------|
| `Country name` | Country |
| `Regional indicator` | Geographic region |
| `Ladder score` | Happiness score (0-10 Cantril ladder) |
| `Logged GDP per capita` | Natural log of GDP per capita |
| `Social support` | Social support index (0-1) |
| `Healthy life expectancy` | Healthy life expectancy at birth (years) |
| `Freedom to make life choices` | Freedom index (0-1) |
| `Generosity` | Generosity score |
| `Perceptions of corruption` | Corruption perception index (0-1) |

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Programming language |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Core plotting library |
| Seaborn | Statistical visualizations |
| Jupyter Notebook | Interactive analysis environment |
| Pathlib | Portable file path handling |

---

## Project Structure

```
Task-1-World-Happiness-Analysis/
├── README.md                    <- This file
├── data/
│   ├── README.md                <- Dataset documentation
│   └── world_happiness_2023.csv <- Dataset (WHR 2023)
├── notebooks/
│   └── world_happiness_eda.ipynb <- Jupyter notebook
├── src/
│   └── world_happiness_analysis.py <- Main Python script
└── outputs/
    ├── figures/                 <- All saved plots (PNG)
    └── reports/                 <- Text findings report
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone or download the project
cd "AVIRENZA-Technologies"

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run the Python Script

```bash
# Navigate to the Task 1 folder
cd "Task-1-World-Happiness-Analysis"

# Run the analysis script
python src/world_happiness_analysis.py
```

**Expected Output:**
- Dataset inspection printed to console
- Data cleaning steps documented
- Descriptive statistics printed
- 6 figures saved to `outputs/figures/`
- Text report saved to `outputs/reports/eda_findings.txt`

---

## How to Run the Jupyter Notebook

```bash
# Navigate to the Task 1 folder
cd "Task-1-World-Happiness-Analysis"

# Launch Jupyter
jupyter notebook notebooks/world_happiness_eda.ipynb
```

Run all cells sequentially from top to bottom (Cell > Run All).

---

## Data Cleaning Methodology

| Step | Action | Rationale |
|------|--------|-----------|
| 1 | Strip whitespace from column names | Prevents key-matching errors |
| 2 | Remove exact duplicate rows | Avoids double-counting |
| 3 | Handle missing values (median fill) | Median is robust to outliers; no rows deleted unnecessarily |
| 4 | Validate Ladder score range (0–10) | Cantril ladder is bounded; out-of-range values signal data errors |
| 5 | Coerce numeric columns to float64 | Ensures correct arithmetic and statistics |
| 6 | Reset index | Consistent indexing after any row removal |

**Result:** Dataset had 0 missing values and 0 duplicates — no rows were removed.

---

## EDA Methodology

1. **Univariate Analysis** – Distribution of happiness scores (histogram + KDE)
2. **Bivariate Analysis** – Happiness vs GDP, Social Support (scatter plots with trend)
3. **Multivariate Analysis** – Correlation heatmap of all numeric variables
4. **Comparative Analysis** – Top/bottom country rankings, regional boxplots
5. **Correlation Analysis** – Pearson correlation matrix with strength/direction labels

---

## Visualizations

| File | Description |
|------|-------------|
| `happiness_distribution.png` | Histogram + KDE of happiness scores with mean/median lines |
| `top_bottom_happiness.png` | Horizontal bar chart: Top 10 and Bottom 10 countries |
| `happiness_vs_gdp.png` | Scatter plot of happiness vs GDP with regression trendline |
| `correlation_heatmap.png` | Pearson correlation heatmap (lower triangle) |
| `regional_happiness_boxplot.png` | Boxplot of happiness by geographic region |
| `social_support_vs_happiness.png` | Regression scatter plot: social support vs happiness |

---

## Key Findings (Derived from Dataset)

> All findings below are computed directly from the WHR 2023 data.

1. **Happiness Range:** Scores range from **2.852** (Mozambique) to **7.804** (Finland).
   Global mean = **5.608**, median = **5.508**, std dev = **1.097**.

2. **Top 5 Countries:** Finland (7.804), Denmark (7.586), Iceland (7.530),
   Israel (7.473), Netherlands (7.403).

3. **Bottom 5 Countries:** Mozambique (2.852), Botswana (3.572), Rwanda (3.703),
   Zimbabwe (3.802), Ukraine (3.846).

4. **Strongest Positive Correlates with Happiness:**
   - Freedom to make life choices: r = +0.910
   - Logged GDP per capita: r = +0.898
   - Social support: r = +0.881
   - Healthy life expectancy: r = +0.827

5. **Negative Correlate:** Perceptions of corruption: r = -0.735
   (Higher corruption perception is associated with lower happiness.)

6. **Regional Pattern:** North America & ANZ region (7.018 avg) scores highest;
   Sub-Saharan Africa (4.165 avg) scores lowest.

7. **Generosity** shows the weakest association with happiness (r = -0.162),
   suggesting it is not a strong linear predictor in this dataset.

> **Note:** Correlation does NOT imply causation. These findings are descriptive and
> suitable for further hypothesis-driven research.

---

## Limitations

- Dataset is cross-sectional (one year only); temporal trends are not captured.
- Self-reported life satisfaction may contain cultural response bias.
- Some small or conflict-affected nations may be missing.
- Correlation analysis assumes linear relationships; non-linear patterns may exist.
- Confounding variables (e.g., inequality within countries) are not fully captured.

---

## Future Improvements

- Merge multiple years (2015–2023) for trend analysis.
- Apply regression or machine learning to model happiness scores.
- Include additional indicators (inequality, political stability).
- Perform clustering to group countries by happiness profile.
- Build an interactive dashboard using Plotly or Streamlit.

---

## License

This project is created for educational purposes as part of the AVIRENZA Technologies AI/ML Internship.
