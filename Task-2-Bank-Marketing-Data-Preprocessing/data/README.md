# UCI Bank Marketing Dataset Documentation

## Dataset Overview

The **Bank Marketing Dataset** is sourced from the **UCI Machine Learning Repository** (originally provided by S. Moro, P. Cortez, and P. Rita, 2014). It contains real data associated with direct marketing campaigns (via phone calls) of a Portuguese banking institution. The marketing campaigns aimed to predict whether a client would subscribe to a bank term deposit (`y`).

---

## Dataset Details

| Attribute | Details |
|---|---|
| **Source** | [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Primary File Used** | `bank-full.csv` (complete dataset) |
| **Secondary / Sample File** | `bank.csv` (10% sample, 4,521 rows) |
| **Total Instances (Rows)** | 45,211 |
| **Total Features (Columns)** | 16 input features + 1 target variable (`y`) |
| **Format / Delimiter** | Semicolon-separated values (`;`) |
| **Target Variable** | `y` (`yes` = subscribed, `no` = did not subscribe) |
| **Class Distribution** | `no`: 39,922 (88.30%), `yes`: 5,289 (11.70%) |

---

## Feature Descriptions

### 1. Bank Client Data
- **`age`** (*numeric*): Age of the client.
- **`job`** (*categorical*): Type of job (`admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`).
- **`marital`** (*categorical*): Marital status (`divorced`, `married`, `single`; note: `divorced` includes widowed).
- **`education`** (*categorical*): Level of education (`primary`, `secondary`, `tertiary`, `unknown`).
- **`default`** (*categorical*): Has credit in default? (`no`, `yes`).
- **`balance`** (*numeric*): Average yearly balance, in euros.
- **`housing`** (*categorical*): Has housing loan? (`no`, `yes`).
- **`loan`** (*categorical*): Has personal loan? (`no`, `yes`).

### 2. Current Campaign Contact Details
- **`contact`** (*categorical*): Contact communication type (`cellular`, `telephone`, `unknown`).
- **`month`** (*categorical*): Last contact month of year (`jan`, `feb`, `mar`, ..., `nov`, `dec`).
- **`day`** (*numeric*): Last contact day of the month (1 to 31).
- **`duration`** (*numeric*): Last contact duration, in seconds.

### 3. Other Attributes / Historical Campaign Data
- **`campaign`** (*numeric*): Number of contacts performed during this campaign and for this client (includes last contact).
- **`pdays`** (*numeric*): Number of days that passed by after the client was last contacted from a previous campaign (`-1` means client was not previously contacted).
- **`previous`** (*numeric*): Number of contacts performed before this campaign and for this client.
- **`poutcome`** (*categorical*): Outcome of the previous marketing campaign (`failure`, `nonexistent`, `success`, `unknown`).

### 4. Target Variable
- **`y`** (*binary categorical*): Has the client subscribed a term deposit? (`yes`, `no`).

---

## Missing & Unknown Value Analysis

The raw dataset does not contain standard `NaN` values (`isnull().sum() == 0`). Instead, missing information is explicitly recorded as `"unknown"` in specific categorical features:

- `poutcome`: 36,959 instances (81.75% unknown - client was not contacted before or outcome not recorded)
- `contact`: 13,020 instances (28.80% unknown - communication method not recorded)
- `education`: 1,857 instances (4.11% unknown)
- `job`: 288 instances (0.64% unknown)

### Preprocessing Strategy:
1. `"unknown"` values in high-frequency context fields (`poutcome`, `contact`) represent informative business states (e.g. new clients never contacted previously).
2. For general machine learning pipelines, categorical features are imputed with `strategy="most_frequent"` and one-hot encoded (`handle_unknown="ignore"`), while numerical features are median-imputed and standardized (`StandardScaler`).
3. Outliers in `balance`, `duration`, and `campaign` are preserved as genuine customer wealth and campaign activity distributions.

---

## Git & File Management

- Raw CSV files are included in the repository for convenience and reproducibility.
- Semicolon delimiter parsing is handled automatically using `sep=';'` in pandas.
