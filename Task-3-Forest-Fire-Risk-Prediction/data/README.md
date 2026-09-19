# Forest Fires Dataset Information

## Dataset Overview
- **Dataset Name:** Forest Fires Dataset
- **Repository:** UCI Machine Learning Repository
- **Source Link:** [UCI Forest Fires Dataset](https://archive.ics.uci.edu/dataset/162/forest+fires)
- **Origin & Creators:** Paulo Cortez and Aníbal de Morais, Department of Information Systems / Center for Research and Development, University of Minho, Portugal (2007).
- **Study Location:** Montesinho Natural Park, Trás-os-Montes region, Northeast Portugal.
- **Total Records:** 517 observations
- **Total Attributes:** 13 variables (including spatial coordinates, temporal indicators, FWI indices, weather metrics, and burned area)

---

## Attribute Information

| Variable | Type | Description |
| :--- | :--- | :--- |
| `X` | Integer (1 to 9) | X-axis spatial coordinate within the Montesinho park map |
| `Y` | Integer (2 to 9) | Y-axis spatial coordinate within the Montesinho park map |
| `month` | Categorical | Month of the year: `'jan'` to `'dec'` |
| `day` | Categorical | Day of the week: `'mon'` to `'sun'` |
| `FFMC` | Float | Fine Fuel Moisture Code from the FWI system: $18.7$ to $96.20$ |
| `DMC` | Float | Duff Moisture Code from the FWI system: $1.1$ to $291.3$ |
| `DC` | Float | Drought Code from the FWI system: $7.9$ to $860.6$ |
| `ISI` | Float | Initial Spread Index from the FWI system: $0.0$ to $56.10$ |
| `temp` | Float | Outside ambient temperature in Celsius: $2.2$ to $33.30$ |
| `RH` | Integer / Float | Relative Humidity in percentage: $15.0$ to $100.0$ |
| `wind` | Float | Wind speed in km/h: $0.40$ to $9.40$ |
| `rain` | Float | Outside rain in mm/m²: $0.0$ to $6.4$ |
| `area` | Float | Burned area of the forest (in hectares): $0.00$ to $1090.84$ |

---

## Target Definition & Leakage Prevention

For this supervised classification study, a binary classification target was derived from the continuous `area` column:

$$\text{fire\_occurrence} = \begin{cases} 1, & \text{if } \text{area} > 0 \text{ (Burned area recorded / Fire event)} \\ 0, & \text{if } \text{area} = 0 \text{ (No burned area recorded)} \end{cases}$$

### Class Distribution (After Deduplication):
- **Total Valid Observations:** 513
- **Class 0 (No Fire):** 244 (47.56%)
- **Class 1 (Fire Occurred):** 269 (52.44%)

### Strict Target Leakage Prevention:
The `area` variable is **strictly removed** from the input feature matrix ($X$) before train-test splitting and preprocessing. The feature space consists exclusively of:
- **Numerical Features (10):** `['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']`
- **Categorical Features (2):** `['month', 'day']`

---

## Preprocessing Decisions
1. **Deduplication:** 4 exact duplicate rows across all 13 columns were identified and removed, leaving 513 unique records.
2. **Missing Values:** Zero missing values exist in the raw dataset. However, a defensive `SimpleImputer` (median for numerical, most frequent for categorical) is embedded into the pipeline for production resilience.
3. **Encoding & Scaling:**
   - Numerical features are normalized via `StandardScaler()`.
   - Categorical features (`month`, `day`) are encoded via `OneHotEncoder(handle_unknown='ignore')`.
4. **Data Leakage Isolation:** All scalers, imputers, and encoders are fit **strictly** on the training partition ($X_{\text{train}}$) and only applied (transform) to test data.
