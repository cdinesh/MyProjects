# Clustering & PCA — country prioritization for aid allocation

## Executive summary

**HELP International** has a fixed budget (~$10M) and must prioritize **countries by need** using socio-economic and health indicators. This project uses **unsupervised learning**—**PCA** for dimensionality reduction and **hierarchical clustering** + **K-Means** (elbow/silhouette for **k**)—to form **actionable country segments** for leadership.

## Business questions

- Which countries form **coherent development profiles**?
- Where should marginal aid dollars **reduce the most suffering** per the chosen metrics?
- How do **2D/3D PCA views** support executive storytelling?

## Technical approach (from notebook)

1. **EDA** — Distributions, scaling, correlation understanding on country-level features (`child_mort`, `income`, `life_expec`, trade/GDP ratios, etc.).
2. **PCA** — Explained variance analysis; **4 components** (~97% variance in the documented run); loadings and heatmaps for interpretation.
3. **Hierarchical clustering** — Linkage, dendrogram, `cut_tree` for cluster labels on PCA space and original features.
4. **K-Means** — SSD elbow and **silhouette** for optimal **k**; cluster visualization in PCA space and on selected original axes.

## Repository contents

| File | Role |
|------|------|
| `ClusteringAndPCA.ipynb` | Full workflow |

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sklearn.decomposition.PCA`, `sklearn.cluster.KMeans`
- `scipy.cluster.hierarchy` (linkage, dendrogram, cut_tree)

## How to run

```bash
cd ClusteringAndPCA
jupyter lab  # open ClusteringAndPCA.ipynb
```

Ensure the **countries dataset** path in the notebook matches your local CSV.

## Outcomes for stakeholders

- **Segmentation** that is reproducible and visually explainable.
- A bridge from **raw development indicators** to **prioritized country lists** for planning.

---

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.
