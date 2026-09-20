# Dataset Documentation: Individual Household Electric Power Consumption

**Task 4:** Electricity Consumption Prediction with Feature Engineering and Model Improvement  
**Internship:** AVIRENZA Technologies  
**Student:** Tharani Natarajan | IFET College of Engineering  

---

## 1. Source & Citation

- **Dataset Name:** Individual Household Electric Power Consumption
- **Origin:** UCI Machine Learning Repository
- **URL:** [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Donors:** Georges Hébrail, Alice Bérard (EDF R&D)
- **Location:** Sceaux (suburbs of Paris, France)
- **Duration:** December 2006 to November 2010 (47 months / 4 years)

---

## 2. Preprocessing & Resampling Details

The raw source dataset contains 2,075,259 minute-by-minute measurements. For stable, robust predictive modeling:
1. **Datetime Indexing:** Date (`dd/mm/yyyy`) and Time (`hh:mm:ss`) columns were parsed into a unified UTC-aware `DatetimeIndex`.
2. **Hourly Aggregation:** Minute-level measurements were resampled into continuous hourly averages (`resample('h').mean()`), yielding **34,589 hourly records**.
3. **Missing Value Imputation:** Missing continuous readings (~1.25% of original minute measurements) were imputed via linear temporal interpolation prior to feature generation.

---

## 3. Schema & Feature Definitions

| Column | Type | Unit | Description |
|---|---|---|---|
| `Datetime` | DateTime | YYYY-MM-DD HH:MM:SS | Timestamp index at 1-hour frequency. |
| `Global_active_power` | float64 | kilowatt (kW) | **Target Variable**: Household global active power (mean). |
| `Global_reactive_power` | float64 | kilowatt (kW) | Household global reactive power (mean). |
| `Voltage` | float64 | volt (V) | Average hourly voltage. |
| `Global_intensity` | float64 | ampere (A) | Average hourly current intensity. |
| `Sub_metering_1` | float64 | watt-hour | Active energy for kitchen (dishwasher, microwave). |
| `Sub_metering_2` | float64 | watt-hour | Active energy for laundry room (washing machine, dryer). |
| `Sub_metering_3` | float64 | watt-hour | Active energy for climate control (water heater, AC). |

---

## 4. Leakage Prevention Notice

In accordance with machine learning best practices:
- Features are never computed using concurrent target values.
- Rolling window features are calculated strictly after applying `shift(1)` to target variables.
- Sub-metering features (which are direct additive constituents of active power) are excluded from predictive feature matrices to avoid trivial target leakage.
