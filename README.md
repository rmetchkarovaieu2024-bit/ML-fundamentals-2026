# ML-fundamentals-2026

**Author:** Raya Metchkarova
**Course:** AI — Machine Learning Foundations
---

This repository contains my individual assignments for the Machine Learning Foundations course. Each one builds on the previous — moving from data preparation and classification in Assignment 1 to regression and model tuning in Assignment 2.

---

## Repository Structure

```
ML-fundamentals-2026/
│
├── assignment_1_raya_metchkarova.ipynb   # Classification — Bank Marketing dataset
├── assignment_2_raya_metchkarova.ipynb   # Regression — Bike Sharing dataset
├── bank-additional.csv                   # Dataset for Assignment 1
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## Assignment 1 — Data Preparation & Classification

**Dataset:** UCI Bank Marketing (`bank-additional.csv`) — telemarketing calls from a Portuguese bank
**Goal:** Predict whether a client will subscribe to a term deposit (`y = yes/no`)

### What's covered

- Exploratory data analysis — class distribution, missing values, feature types
- Deliberate split-first pipeline (70/15/15, stratified) to prevent data leakage
- Handling `unknown` values as a valid category rather than imputing or dropping
- Engineering `was_contacted_before` from the `pdays` sentinel value (999 = never contacted)
- OrdinalEncoder for `education`, OneHotEncoder for all nominal features
- StandardScaler fitted on training set only
- Variance + correlation-based feature selection (removed `emp.var.rate` and `nr.employed` — both r > 0.94 with `euribor3m`)
- SMOTE applied to training only to address the 8:1 class imbalance
- Logistic Regression evaluated with Accuracy, Precision, Recall, F1 — benchmarked against a Zero Rule baseline

### How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook assignment_1_raya_metchkarova.ipynb
```

> The dataset (`bank-additional.csv`) is already in the repo root.

---

## Assignment 2 — Feature Engineering & Regression

**Dataset:** UCI Bike Sharing (hourly) — loaded via `ucimlrepo` (id=275)
**Goal:** Predict the total number of bike rentals per hour (`cnt`)

### What's covered

- EDA focused on the hour-of-day commute pattern (dual peak at 8 AM and 5–6 PM), seasonal trends, and weather effects
- Chronological 60/20/20 split — no shuffling, because predicting future demand from past data is the realistic scenario
- Feature engineering: sin/cos cyclical encoding for `hr` and `weekday`, one-hot encoding for `season`/`weathersit`/`mnth`, `temp × hum` interaction term, StandardScaler on continuous features
- Dropped `atemp` — nearly perfectly collinear with `temp` (r ≈ 0.99)
- Log-transformed target `log(1+cnt)` — reduces right skew and improves model fit
- Three models compared: Linear Regression (baseline), Random Forest, Gradient Boosting
- Hyperparameter tuning: RandomizedSearchCV for RF, BayesSearchCV (Bayesian Optimisation) for GB
- Final model (Tuned Gradient Boosting) retrained on train+val combined, evaluated on held-out test set

### How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-optimize ucimlrepo
jupyter notebook assignment_2_raya_metchkarova.ipynb
```

> The dataset is fetched automatically via `ucimlrepo` — no manual download needed.

---

## Dependencies

| Package | Used for |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, models, evaluation |
| `imbalanced-learn` | SMOTE (Assignment 1) |
| `scikit-optimize` | BayesSearchCV (Assignment 2) |
| `ucimlrepo` | Dataset loading (Assignment 2) |

Install everything at once:

```bash
pip install -r requirements.txt
```

---

## Notes on running the notebooks

- Always use **Restart Kernel → Run All** before submitting or sharing — this confirms the notebook runs clean from top to bottom.
- If you're on PyCharm and see `matmul` RuntimeWarnings, add the following at the top of the notebook — they don't affect results but can be noisy:

```python
import warnings, numpy as np
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
np.seterr(over='ignore', divide='ignore', invalid='ignore')
```

---

## AI tools used

- **Claude.ai** — code review against assignment criteria, README drafting, debugging
- **GitHub Copilot** — inline autocomplete, catching syntax errors
- **PyCharm AI** — local code suggestions during development
