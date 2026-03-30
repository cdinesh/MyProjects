# Portfolio — Data Science, ML & AI Engineering

This repository is a **curated collection of end-to-end analytical work**: from **regression and credit risk** to **NLP, marketing attribution, fraud detection, and modern AI systems**. It is structured so hiring managers and technical leaders can quickly see **scope, methods, and business alignment**.

---

## Leadership narrative

| Strength | How it shows up here |
|----------|----------------------|
| **Problem framing** | Each project states a business or research question before methods (pricing, conversion, risk, segmentation, attribution). |
| **Rigorous modeling** | Regularization, VIF/RFE, stratified splits for imbalance, WOE/IV, ROC–PR thinking, and multi-model comparison. |
| **Scale & systems** | BigQuery pipelines, feature-rich NLP, reproducible notebook chains, and deployable patterns (RAG, OCR APIs). |
| **Communication** | EDA, visual storytelling, funnel and channel views, and executive summaries in project READMEs. |

---

## Project index

| Project | One-line value | Primary artifacts |
|---------|----------------|-------------------|
| [**HandsOnAI**](./HandsOnAI/) | First-principles path to transformers, RAG, and OCR | Notebooks + `5.RAG/`, `6.OCR/` apps |
| [**LeadConversion**](./LeadConversion/) | Ads + AI conversations → features, models, attribution | Phased notebooks + [`docs/WORKBOOK.md`](./LeadConversion/docs/WORKBOOK.md) |
| [**RegressionAnalysis**](./RegressionAnalysis/) | Real-estate-style regression with EDA, MLR, and boosted trees | `Group54_analysis.ipynb`, `python/Group54_analysisFinal.ipynb` |
| [**CarPricePrediction**](./CarPricePrediction/) | US car price drivers via linear models, RFE, VIF | `CarPricePrediction.ipynb` |
| [**HousePricePrediction**](./HousePricePrediction/) | Australian housing with ridge/lasso and feature engineering | `HousePriceRegression.ipynb` |
| [**LeadScoring**](./LeadScoring/) | Hot-lead prioritization with logistic regression, RFE, VIF | `LeadScoring.ipynb`, `Leads.csv` |
| [**LoanDefaultPrediction**](./LoanDefaultPrediction/) | Credit default risk; WOE/IV, GLM, RF, XGBoost; optional multi-table join | `LoanDefaultPrediction.ipynb`, `data/` |
| [**MarketingCampaign**](./MarketingCampaign/) | Bank term-deposit response; WOE/IV; logistic, PCA, RF, XGBoost | `MarketingCampaign.ipynb` |
| [**CreditCardFraudDetection**](./CreditCardFraudDetection/) | Highly imbalanced fraud; SMOTE/ADASYN, power transform, tuned classifiers | `CreditCardFraudDetectionCaseStudy.ipynb` |
| [**ClusteringAndPCA**](./ClusteringAndPCA/) | Country aid prioritization via PCA, hierarchical clustering, K-Means | `ClusteringAndPCA.ipynb` |
| [**CrossDeviceAttribution**](./CrossDeviceAttribution/) | Display attribution (Criteo/KDD-style); preprocessing, EDA, decay attribution | `Experiments.ipynb`, Python modules |
| [**RedditPostsClassification**](./RedditPostsClassification/) | Reddit scrape + binary text classification (Anger vs Meditation) | `code_scrape_reddit.ipynb`, `code_cleaning_modeling_prediction.ipynb` |

---

## How to navigate

1. Open a project folder and read its **`README.md`** for objectives, notebook order, data expectations, and stack.
2. For **HandsOnAI** and **LeadConversion**, follow **numbered subfolders** and nested READMEs.
3. Prefer a **fresh virtual environment** per project when dependencies differ.

---

## Tech themes across projects

- **Core:** Python, pandas, numpy, scikit-learn, statsmodels, matplotlib/seaborn  
- **Boosting / imbalance:** XGBoost, LightGBM, imbalanced-learn (SMOTE, etc.)  
- **NLP:** CountVectorizer, lemmatization, TextBlob, custom stopword strategy  
- **Cloud / scale:** Google BigQuery (LeadConversion)  
- **Modern AI:** PyTorch, transformers track, FAISS RAG, FastAPI OCR (HandsOnAI)

---

## Note on data and credentials

Several notebooks expect **local CSVs** or **cloud credentials** (BigQuery, Reddit API). Do not commit secrets, API keys, or sensitive customer data. Project READMEs call out data layout where it matters.

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.

---

*Optional: add your name, role title, LinkedIn, and email for public portfolio use.*
