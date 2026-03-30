# Marketing campaign response — bank term deposit (Portugal)

## Executive summary

A **Portuguese bank** ran **phone campaigns** for **term deposits**. Marketing leadership needs a **response model** to hit **conversion targets under budget**, report **average call duration** for targeted deciles, and estimate **cost of acquisition**. The notebook combines **WOE/IV** ranking, **logistic regression** (including **PCA** variant), **random forest**, and **XGBoost**, with evaluation on **sensitivity** and **ROC-AUC**.

## Business questions

- Who are the **top X%** of prospects to call given a fixed dial budget?
- Which variables **explain subscription** after monotonic binning / WOE treatment?
- What is the **ROI story** (CAC, duration) for the CMO?

## Technical approach (from notebook)

1. **Cleaning & EDA** — 41k+ rows × 21 columns; missingness and distribution checks.
2. **WOE / IV** — Information value for **feature strength** and screening.
3. **Encoding** — `LabelEncoder` for categoricals where specified.
4. **Part I (with `duration`)** — Train/test split; **statsmodels** + **VIF** workflow; logistic with filtered features; **PCA + logistic**; **RF / XGBoost**; train/test metrics and ROC.
5. **Model choice** — Documented preference for **logistic** in the stated run (interpretability + performance trade-off).

## Repository contents

| File | Role |
|------|------|
| `MarketingCampaign.ipynb` | End-to-end analysis |

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `statsmodels`

## How to run

```bash
cd MarketingCampaign
jupyter lab
```

## Outcomes for stakeholders

- **List prioritization** for outbound sales within **compliance-friendly** interpretable models.
- **Campaign economics** hooks (duration, CAC) for executive dashboards.

---

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.
