# Cross-device / display advertising attribution

## Executive summary

This work is anchored in **industry attribution research** (*Attribution Modeling Increases Efficiency of Bidding in Display Advertising*, Diemert et al., **AdKDD / KDD 2017**) and companion **Criteo-style** experimental code. It combines **notebook-driven EDA and modeling** with **Python modules** for **clickstream processing**, **feature construction**, and **Markov-style** channel transition analysis toward **conversion**.

## Business questions

- How should **credit for conversions** be distributed across **touchpoints** in a user journey?
- Can **exponential decay** (or related) attribution parameters be **estimated** and applied at scale?
- How do **offline metrics** support bidding or budget decisions?

## Repository structure

| Asset | Role |
|-------|------|
| `Experiments.ipynb` | Primary notebook: paper context, preprocessing, EDA, labels, train/validation, **exponential-decay attribution**, full-dataset AA attributions |
| `ExperimentsOG.ipynb` | Earlier / alternate experiment variant (use if comparing iterations) |
| `Untitled.ipynb` | Scratch (safe to ignore for reviewers) |
| `preprocessing.py`, `preproces.py` | Data prep utilities (note legacy filename on one module) |
| `feature_engineering.py` | Feature construction for modeling |
| `clickstream.py` | Journey / sequence handling |
| `model_selection.py` | Loads processed features; **transition matrix** over channels; **simulated conversion probability** from Markov structure |
| `DDA.py` | Additional attribution-related logic (see in-file docstrings) |
| `test.py` | Tests or ad-hoc checks |

## Technical themes

- **Preprocessing → EDA → supervised learning** for attribution-related prediction.
- **Attribution model** block learns an **exponential decay λ** (as described in notebook).
- **Transition matrices** and **vectorized simulation** of path-to-conversion probability (`model_selection.py`).

## Dependencies

Typical stack: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` — confirm versions against your interpreter. A **local venv** (`isye/` may exist in your clone) can be recreated cleanly for portability.

## How to run

1. Obtain the **dataset** referenced in the paper/README bundled with the original release (not stored in this portfolio snapshot).
2. Open `Experiments.ipynb` and align **paths** to your data location.
3. Run supporting scripts from the project root so relative paths resolve.

## Outcomes for stakeholders

- Demonstrates familiarity with **ad-tech measurement**, **reproducible research** workflows, and **hybrid notebook + library** engineering.

---

**Note:** Due to storage limits, the dataset is not included in this repository and can be made available upon request.
