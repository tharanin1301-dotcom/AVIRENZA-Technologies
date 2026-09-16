"""
=============================================================================
World Happiness Analysis - Python Script
=============================================================================
Project  : Task 1 – Exploratory Data Analysis on a Real Dataset
Title    : World Happiness Analysis Using Python
Company  : AVIRENZA Technologies
Intern   : Tharani Natarajan
College  : IFET College of Engineering
Dept     : Artificial Intelligence and Data Science
Dataset  : World Happiness Report 2023
Source   : https://worldhappiness.report/data/
=============================================================================
"""

# ---------------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend (script mode)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 2. CONFIGURE PATHS
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
FIG_DIR    = BASE_DIR / "outputs" / "figures"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE  = DATA_DIR / "world_happiness_2023.csv"

# Consistent plot styling
PALETTE    = "viridis"
ACCENT     = "#4C72B0"
BG_COLOR   = "#F8F9FA"
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ---------------------------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------------------------
def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load the World Happiness dataset from a CSV file.
    Returns a pandas DataFrame.

    Expected columns (WHR 2023 format):
        Country name, Regional indicator, Ladder score,
        Logged GDP per capita, Social support,
        Healthy life expectancy, Freedom to make life choices,
        Generosity, Perceptions of corruption
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please place 'world_happiness_2023.csv' in the data/ folder."
        )
    df = pd.read_csv(filepath)
    print(f"[OK] Dataset loaded from: {filepath}")
    return df


# ---------------------------------------------------------------------------
# 4. DATA INSPECTION
# ---------------------------------------------------------------------------
def inspect_data(df: pd.DataFrame) -> None:
    """Print a thorough inspection summary of the dataset."""
    sep = "=" * 65

    print(f"\n{sep}\nDATASET INSPECTION\n{sep}")

    print(f"\n--- Shape: {df.shape[0]} rows x {df.shape[1]} columns ---")

    print("\n--- First 5 Rows ---")
    print(df.head().to_string())

    print("\n--- Last 5 Rows ---")
    print(df.tail().to_string())

    print("\n--- Column Names ---")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")

    print("\n--- Data Types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "  No missing values found.")

    print(f"\n--- Duplicate Rows: {df.duplicated().sum()} ---")

    print("\n--- Descriptive Statistics ---")
    print(df.describe(include="all").round(3).to_string())

    # Identify column types
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    print(f"\n--- Numerical Columns ({len(num_cols)}) ---")
    print(" | ".join(num_cols))
    print(f"\n--- Categorical Columns ({len(cat_cols)}) ---")
    print(" | ".join(cat_cols))

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# 5. DATA CLEANING
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning steps.
    Documents each decision with a rationale.
    Returns a cleaned DataFrame.
    """
    print("=" * 65)
    print("DATA CLEANING")
    print("=" * 65)

    original_shape = df.shape

    # Step 1: Standardise column names (strip whitespace)
    df.columns = df.columns.str.strip()
    print(f"\n[1] Column names stripped of leading/trailing whitespace.")

    # Step 2: Drop exact duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates()
        print(f"[2] Removed {n_dupes} duplicate row(s).")
    else:
        print(f"[2] No duplicate rows found — no action required.")

    # Step 3: Handle missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        print(f"\n[3] Missing values detected:")
        print(missing_cols)
        # Strategy: fill numeric columns with column median (robust to outliers)
        for col in missing_cols.index:
            if df[col].dtype in [np.float64, np.int64, float, int]:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    Filled '{col}' with median ({median_val:.4f})")
            else:
                df[col].fillna("Unknown", inplace=True)
                print(f"    Filled '{col}' with 'Unknown'")
    else:
        print(f"[3] No missing values — no imputation required.")

    # Step 4: Validate numerical ranges
    #   Ladder score: theoretically 0–10 (Cantril ladder)
    #   Proportions (social support, freedom, generosity, corruption): -1 to 1
    ladder_out = df[(df["Ladder score"] < 0) | (df["Ladder score"] > 10)]
    if not ladder_out.empty:
        print(f"\n[4] WARNING: {len(ladder_out)} rows with invalid Ladder score (outside 0-10).")
    else:
        print(f"[4] Ladder score range check passed (all values within 0–10).")

    # Step 5: Ensure correct data types
    num_cols = ["Ladder score","Logged GDP per capita","Social support",
                "Healthy life expectancy","Freedom to make life choices",
                "Generosity","Perceptions of corruption"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[5] Numeric columns coerced to float64 where needed.")

    # Step 6: Reset index after any row removal
    df = df.reset_index(drop=True)
    print(f"[6] Index reset.")

    final_shape = df.shape
    print(f"\n[Summary] Original shape: {original_shape} -> Cleaned shape: {final_shape}")
    print("=" * 65 + "\n")

    return df


# ---------------------------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------
def compute_statistics(df: pd.DataFrame) -> None:
    """Compute and print detailed descriptive statistics."""
    print("=" * 65)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 65)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    stats = df[num_cols].agg(["mean","median","min","max","std"]).T
    stats.columns = ["Mean","Median","Min","Max","Std Dev"]
    print(stats.round(4).to_string())

    print("\n--- Quartiles ---")
    print(df[num_cols].quantile([0.25, 0.50, 0.75]).round(4).to_string())

    print("\n--- Happiness Score Key Facts ---")
    ls = df["Ladder score"]
    print(f"  Highest score : {df.loc[ls.idxmax(), 'Country name']} ({ls.max():.3f})")
    print(f"  Lowest score  : {df.loc[ls.idxmin(), 'Country name']} ({ls.min():.3f})")
    print(f"  Global average: {ls.mean():.3f}")
    print(f"  Std deviation : {ls.std():.3f}")

    if "Regional indicator" in df.columns:
        print("\n--- Average Happiness by Region ---")
        region_avg = (
            df.groupby("Regional indicator")["Ladder score"]
            .mean()
            .sort_values(ascending=False)
            .round(3)
        )
        print(region_avg.to_string())

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# 7. VISUALISATION FUNCTIONS
# ---------------------------------------------------------------------------

def plot_happiness_distribution(df: pd.DataFrame) -> None:
    """
    Visualization 1: Distribution of Happiness (Ladder) Scores.
    Shows a histogram + KDE to reveal the shape and spread of scores.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)

    sns.histplot(
        df["Ladder score"], bins=20, kde=True,
        color=ACCENT, edgecolor="white", linewidth=0.6, ax=ax
    )

    mean_score = df["Ladder score"].mean()
    median_score = df["Ladder score"].median()
    ax.axvline(mean_score,   color="#E74C3C", linestyle="--", linewidth=1.8, label=f"Mean = {mean_score:.2f}")
    ax.axvline(median_score, color="#27AE60", linestyle=":",  linewidth=1.8, label=f"Median = {median_score:.2f}")

    ax.set_title("Distribution of World Happiness Scores (2023)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Happiness (Ladder) Score",  fontsize=12)
    ax.set_ylabel("Number of Countries", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "happiness_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_top_bottom_countries(df: pd.DataFrame, n: int = 10) -> None:
    """
    Visualization 2: Top N and Bottom N countries by Happiness Score.
    Horizontal bar chart for easy comparison.
    """
    top_n    = df.nlargest(n,  "Ladder score")[["Country name","Ladder score"]].iloc[::-1]
    bottom_n = df.nsmallest(n, "Ladder score")[["Country name","Ladder score"]]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f"Top {n} and Bottom {n} Countries by Happiness Score (2023)",
                 fontsize=14, fontweight="bold", y=1.01)

    # Top N
    bars1 = axes[0].barh(
        top_n["Country name"], top_n["Ladder score"],
        color=sns.color_palette("Blues_d", n), edgecolor="white"
    )
    for bar, val in zip(bars1, top_n["Ladder score"]):
        axes[0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                     f"{val:.2f}", va="center", fontsize=9)
    axes[0].set_title(f"Top {n} Happiest Countries", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Happiness Score", fontsize=11)
    axes[0].set_xlim(0, df["Ladder score"].max() + 0.8)
    axes[0].grid(axis="x", alpha=0.3)

    # Bottom N
    bars2 = axes[1].barh(
        bottom_n["Country name"], bottom_n["Ladder score"],
        color=sns.color_palette("Reds_d", n), edgecolor="white"
    )
    for bar, val in zip(bars2, bottom_n["Ladder score"]):
        axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                     f"{val:.2f}", va="center", fontsize=9)
    axes[1].set_title(f"Bottom {n} Least Happy Countries", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Happiness Score", fontsize=11)
    axes[1].set_xlim(0, df["Ladder score"].max() + 0.8)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "top_bottom_happiness.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_happiness_vs_gdp(df: pd.DataFrame) -> None:
    """
    Visualization 3: Happiness Score vs Logged GDP per Capita.
    Scatter plot with regression trendline and region colour coding.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG_COLOR)

    regions = df["Regional indicator"].unique() if "Regional indicator" in df.columns else ["All"]
    palette = sns.color_palette("tab10", len(regions))

    if "Regional indicator" in df.columns:
        for region, color in zip(regions, palette):
            mask = df["Regional indicator"] == region
            ax.scatter(
                df.loc[mask, "Logged GDP per capita"],
                df.loc[mask, "Ladder score"],
                label=region, color=color, alpha=0.75, s=65, edgecolor="white", linewidth=0.5
            )
    else:
        ax.scatter(df["Logged GDP per capita"], df["Ladder score"], color=ACCENT, alpha=0.75, s=65)

    # Regression trendline using numpy polyfit
    x_vals = df["Logged GDP per capita"].dropna()
    y_vals = df.loc[x_vals.index, "Ladder score"]
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#E74C3C", linewidth=2,
            linestyle="--", label=f"Trend (slope={m:.2f})")

    # Annotate a few notable countries
    for _, row in df.nlargest(3, "Ladder score").iterrows():
        ax.annotate(row["Country name"],
                    (row["Logged GDP per capita"], row["Ladder score"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8, color="#333")
    for _, row in df.nsmallest(3, "Ladder score").iterrows():
        ax.annotate(row["Country name"],
                    (row["Logged GDP per capita"], row["Ladder score"]),
                    xytext=(4, -10), textcoords="offset points", fontsize=8, color="#333")

    ax.set_title("Happiness Score vs Logged GDP per Capita (2023)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Logged GDP per Capita", fontsize=12)
    ax.set_ylabel("Happiness (Ladder) Score", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "happiness_vs_gdp.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Visualization 4: Pearson Correlation Heatmap of numerical variables.
    Reveals linear associations between happiness and all indicators.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = df[num_cols].corr(method="pearson")

    # Friendly column labels
    labels = {
        "Ladder score": "Happiness",
        "Logged GDP per capita": "GDP/capita",
        "Social support": "Social Support",
        "Healthy life expectancy": "Life Expectancy",
        "Freedom to make life choices": "Freedom",
        "Generosity": "Generosity",
        "Perceptions of corruption": "Corruption",
    }
    corr_matrix = corr_matrix.rename(index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG_COLOR)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))   # upper triangle mask
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        mask=mask, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}, ax=ax
    )

    ax.set_title("Pearson Correlation Heatmap – World Happiness 2023",
                 fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    path = FIG_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_regional_boxplot(df: pd.DataFrame) -> None:
    """
    Visualization 5 (optional): Boxplot of Happiness Score by Region.
    Shows median, spread, and outliers across geographic regions.
    """
    if "Regional indicator" not in df.columns:
        print("[Skip] No 'Regional indicator' column found for regional plot.")
        return

    # Sort regions by median happiness for cleaner display
    order = (
        df.groupby("Regional indicator")["Ladder score"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(BG_COLOR)

    sns.boxplot(
        data=df, x="Ladder score", y="Regional indicator",
        order=order, palette="viridis", width=0.55,
        linewidth=1.2, fliersize=4, ax=ax
    )

    ax.set_title("Happiness Score by Region (2023)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Happiness (Ladder) Score", fontsize=12)
    ax.set_ylabel("Region", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "regional_happiness_boxplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_social_support_vs_happiness(df: pd.DataFrame) -> None:
    """
    Visualization 6 (optional): Social Support vs Happiness.
    Social support is one of the strongest predictors of happiness.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)

    sns.regplot(
        data=df, x="Social support", y="Ladder score",
        scatter_kws={"alpha": 0.65, "color": "#9B59B6", "s": 60, "edgecolor": "white"},
        line_kws={"color": "#E74C3C", "linewidth": 2},
        ax=ax
    )

    ax.set_title("Social Support vs Happiness Score (2023)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Social Support Score", fontsize=12)
    ax.set_ylabel("Happiness (Ladder) Score", fontsize=12)
    ax.grid(alpha=0.3)

    corr = df[["Social support","Ladder score"]].corr().iloc[0,1]
    ax.text(0.05, 0.93, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    path = FIG_DIR / "social_support_vs_happiness.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ---------------------------------------------------------------------------
# 8. CORRELATION ANALYSIS
# ---------------------------------------------------------------------------
def correlation_analysis(df: pd.DataFrame) -> None:
    """
    Compute and display Pearson correlations with the Happiness score.
    Clearly states that correlation != causation.
    """
    print("=" * 65)
    print("CORRELATION ANALYSIS (Pearson)")
    print("=" * 65)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols_no_target = [c for c in num_cols if c != "Ladder score"]

    corr_with_happiness = (
        df[num_cols]
        .corr(method="pearson")["Ladder score"]
        .drop("Ladder score")
        .sort_values(ascending=False)
    )

    print("\nCorrelation with Happiness (Ladder) Score:")
    for var, val in corr_with_happiness.items():
        direction = "Positive" if val >= 0 else "Negative"
        strength  = ("Strong" if abs(val) >= 0.6
                     else "Moderate" if abs(val) >= 0.4
                     else "Weak")
        print(f"  {var:<35} r = {val:+.4f}  ({strength} {direction})")

    print("\n[!] IMPORTANT: Correlation indicates association, NOT causation.")
    print("    These relationships should inform further research, not policy claims.")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# 9. KEY FINDINGS REPORT
# ---------------------------------------------------------------------------
def print_findings(df: pd.DataFrame) -> None:
    """Print evidence-based findings derived from the dataset."""
    print("=" * 65)
    print("KEY FINDINGS")
    print("=" * 65)

    ls = df["Ladder score"]
    top5    = df.nlargest(5,  "Ladder score")[["Country name","Ladder score"]]
    bottom5 = df.nsmallest(5, "Ladder score")[["Country name","Ladder score"]]

    print(f"\n1. Global Happiness Range: {ls.min():.3f} (lowest) to {ls.max():.3f} (highest)")
    print(f"   Mean = {ls.mean():.3f}, Median = {ls.median():.3f}, StdDev = {ls.std():.3f}")

    print("\n2. Top 5 Happiest Countries:")
    for _, r in top5.iterrows():
        print(f"   {r['Country name']:<40} {r['Ladder score']:.3f}")

    print("\n3. Bottom 5 Least Happy Countries:")
    for _, r in bottom5.iterrows():
        print(f"   {r['Country name']:<40} {r['Ladder score']:.3f}")

    # Strongest correlates
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()["Ladder score"].drop("Ladder score").sort_values(ascending=False)
    print(f"\n4. Strongest positive correlate: '{corr.idxmax()}' (r = {corr.max():+.3f})")
    print(f"   Weakest / most negative:       '{corr.idxmin()}' (r = {corr.min():+.3f})")

    if "Regional indicator" in df.columns:
        region_avg = df.groupby("Regional indicator")["Ladder score"].mean().sort_values(ascending=False)
        print(f"\n5. Happiest region on average    : {region_avg.index[0]} ({region_avg.iloc[0]:.3f})")
        print(f"   Least happy region on average  : {region_avg.index[-1]} ({region_avg.iloc[-1]:.3f})")

    print("\n6. Limitations:")
    print("   • Dataset reflects self-reported life satisfaction; subjective bias may exist.")
    print("   • Cross-sectional data for one year; trends may differ over time.")
    print("   • Correlation != causation; no causal claims are made.")
    print("   • Some smaller nations may be absent from the dataset.")

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# 10. SAVE TEXT REPORT
# ---------------------------------------------------------------------------
def save_report(df: pd.DataFrame) -> None:
    """Save a plain-text findings report to outputs/reports/."""
    import io, sys

    report_path = REPORT_DIR / "eda_findings.txt"

    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    print("WORLD HAPPINESS ANALYSIS – EDA FINDINGS REPORT")
    print("AVIRENZA Technologies | Tharani Natarajan | WHR 2023")
    print("=" * 65)
    compute_statistics(df)
    correlation_analysis(df)
    print_findings(df)

    sys.stdout = old_stdout
    report_path.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"[Saved] Text report -> {report_path}")


# ---------------------------------------------------------------------------
# 11. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 65)
    print("  WORLD HAPPINESS EDA — AVIRENZA TECHNOLOGIES INTERNSHIP")
    print("  Intern: Tharani Natarajan | WHR 2023")
    print("=" * 65 + "\n")

    # Load
    df_raw = load_data(DATA_FILE)

    # Inspect
    inspect_data(df_raw)

    # Clean
    df = clean_data(df_raw)

    # Statistics
    compute_statistics(df)

    # Visualisations
    print("Generating visualisations …\n")
    plot_happiness_distribution(df)
    plot_top_bottom_countries(df)
    plot_happiness_vs_gdp(df)
    plot_correlation_heatmap(df)
    plot_regional_boxplot(df)
    plot_social_support_vs_happiness(df)

    # Correlation Analysis
    correlation_analysis(df)

    # Findings
    print_findings(df)

    # Save report
    save_report(df)

    print("\n[Done] All outputs saved to:")
    print(f"  Figures -> {FIG_DIR}")
    print(f"  Reports -> {REPORT_DIR}\n")


if __name__ == "__main__":
    main()

