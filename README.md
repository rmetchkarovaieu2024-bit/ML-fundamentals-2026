# ML-fundamentals-2026

Machine Learning Foundations — Individual Assignment 1  
**Author:** Raya Metchkarova  
**Course:** AI: Machine Learning Foundations  
**Deadline:** 4 March 2026

---

## Overview

This repository contains the first individual assignment for the Machine Learning Foundations course. The assignment focuses on **data preparation and feature engineering** using the [UCI Bank Marketing dataset](https://www.kaggle.com/) (`bank-additional.csv`) — a real-world dataset from a Portuguese banking institution's telemarketing campaigns.

---

## Repository Structure

```
ML-fundamentals-2026/
│
├── assignment_1_raya_metchkarova.ipynb   # Main notebook — all tasks implemented here
└── README.md                             # This file
```

> **Note:** The dataset (`bank-additional.csv`) is included in this repository, but I don't know if I have rights or not, sooo. Download it from Kaggle and place it in the same directory as the notebook before running.

---

## Dataset

**Source:** UCI Bank Marketing Dataset (smaller version — `bank-additional.csv`)  
**Size:** ~4,100 rows × 21 columns  
**Separator:** semicolon (`;`)  
**Target variable:** `y` — whether the client subscribed to a term deposit (`yes` / `no`)

The dataset includes demographic attributes (age, job, marital status, education), financial attributes (housing loan, personal loan, credit default), campaign-related attributes (contact type, number of contacts, call duration, days since last contact), and macroeconomic indicators (Euribor rate, employment variation rate, consumer price index).

---

## Pipeline Tasks

The notebook implements all required tasks in a deliberate order designed to prevent data leakage. Tasks in the assignment were listed alphabetically — the chosen execution order is:

| Step | Task | Key decision |
|------|------|-------------|
| 1 | **Identify Prediction Target** | `y` is the target; `duration` and `poutcome` discussed as invalid alternatives |
| 2 | **Data Loading & Exploration** | Structure inspection, class distribution, explicit and implicit missing values, visualisations |
| 3 | **Data Splitting** | 70 / 15 / 15 train–val–test split with stratification on `y` |
| 4 | **Manage Missing Values** | `unknown` retained as a category; `pdays=999` converted to binary `was_contacted_before` |
| 5 | **Encode Categorical Variables** | `OrdinalEncoder` for `education`; `OneHotEncoder` (drop first) for all nominal variables |
| 6 | **Feature Scaling** | `StandardScaler` fitted on training set only |
| 7 | **Feature Selection** | Variance threshold (< 0.01); correlation filter (|r| > 0.85) removes `emp.var.rate` and `nr.employed` |
| 8 | **Address Class Imbalance** | SMOTE applied to training set only; val/test preserve original ~11% positive rate |
| 9 | **Train Logistic Regression** | Evaluated with Accuracy, Precision, Recall, F1; compared against Zero Rule baseline |

---

## Key Design Decisions

**Why split before preprocessing?**  
Any transformer that learns parameters from data (scaler, encoder, imputer) must be fitted on the training set only. Splitting first guarantees that validation and test sets are truly unseen during all fitting steps, preventing distribution leakage.

**Why retain `unknown` as a category?**  
Dropping rows with `unknown` values would remove up to ~20% of the data. Mode imputation would invent information that does not exist. Treating `unknown` as a separate category preserves all rows and lets the model learn whether missingness itself is predictive.

**Why engineer `was_contacted_before` from `pdays`?**  
96.3% of `pdays` values are 999 (a sentinel meaning "not previously contacted"). Treating this as a numeric feature creates extreme skew. A binary indicator cleanly captures the meaningful information without any learned parameters.

**Why StandardScaler over MinMaxScaler?**  
Several features (e.g., `age`) contain outliers. `MinMaxScaler` is sensitive to extremes; `StandardScaler` is more robust. Standardisation is also the conventional choice for L2-regularised Logistic Regression, ensuring coefficients are comparable and regularisation is applied fairly across all features.

**Why remove `emp.var.rate` and `nr.employed`?**  
Both are correlated with `euribor3m` at r > 0.94. All three track the same macroeconomic cycle. Retaining all three causes multicollinearity — unstable, uninterpretable coefficients in Logistic Regression. `euribor3m` is kept as the most direct financial predictor for a term deposit product.

**Why SMOTE on training only?**  
The 8:1 class imbalance suppresses recall on the minority class. SMOTE generates synthetic minority points by interpolating between real training examples. Applying it to training only ensures the val/test sets preserve the real-world class distribution, so evaluation metrics are meaningful and unbiased.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ML-fundamentals-2026
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

3. Place `bank-additional.csv` in the repository root directory.

4. Open and run the notebook:
   ```bash
   jupyter notebook assignment_1_raya_metchkarova.ipynb
   ```

5. To fully validate the notebook, use **Restart Kernel and Run All** in Jupyter and confirm every cell executes without errors.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, encoding, scaling, feature selection, Logistic Regression, evaluation metrics |
| `imbalanced-learn` | SMOTE resampling |

---

## AI
This project was used with the help of AI:
- **PyCharm** - AI-powered local code completion; helped when i make a coment
- **GitHub Copilot** - AI-assistant developed by GitHub and OpenAI that assists users of Visual Studio Code, Visual Studio, Neovim, Eclipse and JetBrains integrated development environments by autocompleting code.; Used when there were fundamental mistakes in the code, or there was an explanation needed
- **Claud.ai** - AI with massive context window, enabling deep analysis of long documents; Used for analyzing my code based on the provided criteria and with the help of this READ_ME
- 

---
## PyCharm 
If the code is executed on PyCharm, there may be errors/warnings; VSCode ignores them. The code is still executable. 
Add this to the beginning of the code: 
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*matmul.*",
        category=RuntimeWarning
    )
    model.fit(X_train_res, y_train_res)

np.seterr(over='ignore', divide='ignore', invalid='ignore')


