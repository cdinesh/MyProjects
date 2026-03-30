# Credit card fraud detection

## Executive summary

**Fraud is rare and costly to miss.** This case study builds a **disciplined imbalanced-classification** workflow: EDA on anonymized transaction features, **duplicate-row policy**, **power transforms**, **RFE** with logistic regression for feature screening, and **multiple classifiers** tuned via **RandomizedSearchCV** with **ROC-AUC**—plus **threshold** analysis (precision–recall, ROC).

## Business questions

- How do we **maximize detection of fraud** without collapsing precision on the majority class?
- Which **feature transformations** stabilize separability for V1–V28-style PCA components and amounts/time?
- How do **Logistic Regression, Decision Tree, Random Forest, and XGBoost** compare under strict stratified evaluation?

## Technical approach (from notebook)

1. **EDA** — Class imbalance characterization; duplicate analysis (1k+ duplicate rows noted in notebook narrative).
2. **Split** — **StratifiedKFold**-style train/test construction to preserve rare-class prevalence.
3. **Preprocessing** — `PowerTransformer` on features (as implemented).
4. **Feature selection** — **RFE** with balanced logistic regression.
5. **Models** — Pipelines + **RandomizedSearchCV** (`scoring='roc_auc'`); **class_weight='balanced'`** where used; optional **SMOTE / RandomOverSampler / ADASYN** from `imblearn` in the experimentation flow.
6. **Evaluation** — Confusion matrices, ROC, precision–recall curves, **probability cutoffs**, consolidated results table.

## Repository contents

| File | Role |
|------|------|
| `CreditCardFraudDetectionCaseStudy.ipynb` | Full pipeline |

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `imblearn` (SMOTE, ADASYN, RandomOverSampler)
- `xgboost` (if running XGBoost sections)

## How to run

```bash
cd CreditCardFraudDetection
jupyter lab
```

**Note:** Use a machine with sufficient RAM; training shapes in the notebook are large (hundreds of thousands of rows × tens of features).

## Outcomes for stakeholders

- A **risk-appropriate** framing: accuracy is misleading; **recall/precision at chosen operating point** matters.
- Reusable pattern for **imbalanced retail/finance** use cases.

---

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.
