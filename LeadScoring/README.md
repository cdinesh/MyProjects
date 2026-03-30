# Lead scoring — education vertical

## Executive summary

**X Education** generates many **web leads** but a weak **conversion rate**. The ask is a **lead score** so **sales and marketing** focus on **hot leads** first, with leadership guidance around **~80%** of conversions captured in the prioritized segment. The notebook implements **logistic regression** with **statsmodels** inference, **RFE**, and **VIF** to produce a **defensible, interpretable** scoring foundation.

## Business questions

- Which **behaviors and attributes** precede conversion?
- What is a **ranked list** of leads aligned to revenue capacity?
- Are coefficients **stable** (low multicollinearity, sensible p-values)?

## Technical approach (from notebook)

1. **EDA & cleaning** — Imputation (`SimpleImputer`, `KNNImputer` where used), univariate and bivariate analysis.
2. **Train/test split** — Stratified or standard split per notebook cells.
3. **Feature selection** — **RFE** (top 15 features in documented flow); iterative refinement.
4. **Inference** — **statsmodels** logistic; **VIF** tables; drop/rebuild cycles until acceptable VIF and significance.
5. **Evaluation** — Confusion matrix, accuracy, precision/recall/F1, **ROC-AUC**, ROC and precision–recall curves, **lead scoring** on holdout.

## Repository contents

| File | Role |
|------|------|
| `LeadScoring.ipynb` | Main analysis |
| `Leads.csv` | Lead-level dataset |

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `statsmodels`

## How to run

```bash
cd LeadScoring
jupyter lab  # open LeadScoring.ipynb
```

Ensure `Leads.csv` sits in the working directory expected by the notebook.

## Outcomes for stakeholders

- **Operationalizable ranking** for CRM or dialer integration (export scores from final cells).
- **Audit trail** via statistical tables suitable for **revenue ops** review.

---

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.
