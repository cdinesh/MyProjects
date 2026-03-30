# Car price prediction (US market survey data)

## Executive summary

Consulting-style case: an automaker entering the **US market** needs to understand **which vehicle attributes explain price** and how well a linear model captures that structure. This notebook delivers **interpretable regression**, **multicollinearity control**, and **feature selection** suitable for stakeholder review.

## Business questions

- Which variables are **statistically and practically** significant for price?
- How strong is the **explanatory model** (adjusted R², diagnostics)?
- What is the **minimum viable feature set** after addressing VIF and significance?

## Technical approach (from notebook)

1. **Data understanding** — Full data dictionary; correct dtypes (e.g. insurance risk `symboling` as categorical).
2. **Preparation** — Derived features, encoding, outlier and relationship checks.
3. **Modeling** — `LinearRegression` with **RFE**; **statsmodels** OLS-style inference; **VIF** for multicollinearity; iterative refinement (drop high-VIF / low–p-value features where justified).
4. **Validation mindset** — Residual and significance interpretation (as implemented in the notebook flow).

## Repository contents

| File | Role |
|------|------|
| `CarPricePrediction.ipynb` | End-to-end analysis |

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (e.g. `LinearRegression`, `RFE`)
- `statsmodels` (inference, VIF)

Install as needed for your environment; pin versions for reproducibility if publishing.

## How to run

```bash
cd CarPricePrediction
jupyter lab  # open CarPricePrediction.ipynb
```

Use **Kernel → Restart & Run All** after ensuring the dataset path in the notebook matches your file location.

## Outcomes for stakeholders

- A **transparent** pricing model narrative: drivers of price, trade-offs from collinearity, and a defensible feature list.
- A template for **“consulting deliverable”** regression with statistical hygiene.

---

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.
