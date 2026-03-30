# Loan default prediction — credit risk

## Executive summary

An **Asia-Pacific bank** wants to move beyond static ratings to **data-driven default prediction**: installment payment ability, **default risk**, and (in the problem framing) **monthly contribution** views. The notebook implements **WOE/IV** screening, **stratified** sampling for imbalance, **RFE**, **statsmodels GLM (Binomial)**, **Random Forest**, and **XGBoost**, with a **Part II** that enriches features using **bureau**, **previous applications**, and **installments** when those files are available.

## Business questions

- Which applicants are **likely to default** or miss installments?
- What is the **expected credit loss** vs a baseline without the model?
- Which features are **too strong** (data quality / leakage checks) vs genuinely predictive?

## Technical approach (from notebook)

**Part I — `application_train.csv` only**

- Clean high-missing columns; EDA; **WOE & IV** for predictor strength tiers.
- **StratifiedKFold** train/test; scaling; **RFE**; iterative **VIF** and **GLM** refits with feature drops.
- **RandomizedSearchCV**-style tuning for **RandomForest** and **XGBClassifier**; metric panel (accuracy, specificity, ROC-AUC, etc.).
- Select champion model (documented as **Random Forest** in one branch of the narrative).

**Part II — multi-table**

- Join **previous_applications**, **bureau**, **installments** (as implemented); aggregated features; similar modeling and **business impact** view (credit loss saved vs revenue loss without model).

## Repository contents

| File / folder | Role |
|---------------|------|
| `LoanDefaultPrediction.ipynb` | Full workflow |
| `data/` | `application_train.csv` and related files (as used in notebook) |

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `statsmodels`

## How to run

```bash
cd LoanDefaultPrediction
jupyter lab
```

Populate `data/` per the notebook’s expected filenames before **Restart & Run All**.

## Outcomes for stakeholders

- **Risk-grade** narrative: IV tiers, calibrated probabilities, and champion-model trade-offs.
- Path to **portfolio monitoring** (drift, recalibration) for a second-phase roadmap.

---

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.
