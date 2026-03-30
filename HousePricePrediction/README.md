# House price prediction — regularized regression (Australian market)

## Executive summary

**Surprise Housing** expands into **Australia** and needs models to **estimate fair property value** for buy-and-flip decisions. The notebook builds a **regularized regression** workflow (ridge/lasso) with **log-transformed target**, rich **EDA**, **imputation**, **feature engineering**, **encoding**, and **scaling**—aligned with how analytics teams support **investment committees**.

## Business questions

- Which variables **drive price** in this market?
- What is the **optimal λ** for ridge and lasso (bias–variance and interpretability trade-offs)?
- Is the model **stable** after handling skew, outliers, and categorical structure?

## Technical approach (from notebook)

1. **Target** — Assess skewness; **log transform** of price where appropriate.
2. **EDA** — Outliers, correlations, missingness patterns (e.g. `LotFrontage`, basement/garage features).
3. **Feature engineering** — Domain-motivated fields (living area ratios, age, porch, bathrooms, etc., per notebook).
4. **Encoding** — Ordinal `LabelEncoder`; **dummy** variables for nominal categories; `MSSubClass` as categorical per data dictionary.
5. **Scaling** — `StandardScaler` on selected numeric features before regularized linear models.
6. **Modeling** — **Ridge/Lasso** with cross-validation for **lambda** selection; coefficient interpretation for business narrative.

## Repository contents

| File | Role |
|------|------|
| `HousePriceRegression.ipynb` | End-to-end analysis |

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (preprocessing, ridge/lasso, CV)

## Data

Expect the **Kaggle-style Ames / Australian sale** CSV referenced inside the notebook; place it per the path used in **Load Data** cells.

## How to run

```bash
cd HousePricePrediction
jupyter lab  # open HousePriceRegression.ipynb
```

## Outcomes for stakeholders

- Clear **pricing drivers** and **regularization narrative** for general managers and risk teams.
- Reusable **feature-engineering checklist** for tabular real-estate ML.

---

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.
