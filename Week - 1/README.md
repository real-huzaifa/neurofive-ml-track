# Week 1 — Titanic: Data Cleaning, EDA & Visualization

First week into the NeuroFive ML track. The goal for Week 1 is to take a raw,
messy dataset and carry it through the full early-stage data workflow: loading,
exploration, handling missing values, outlier detection, and visual analysis to
surface what actually drives the target.

---

## Dataset

`titanic_train.csv` — 891 passengers, 12 columns. Target is `Survived`
(0 = did not survive, 1 = survived).

| Type | Columns |
|------|---------|
| Numerical | PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare |
| Categorical | Name, Sex, Ticket, Cabin, Embarked |

---

## Files

| File | Contents |
|------|----------|
| `Titanic - Cleaning And Visualizing.ipynb` | Full workflow: loading → exploration → missing-value handling → outlier detection → visualizations → correlation analysis. **This is the complete notebook.** |
| `Titanic - EDA.ipynb` | Loading and initial exploration only (a subset of the notebook above). |
| `titanic_train.csv` | Raw training dataset. |

---

## Workflow

**1. Exploration**
`.info()`, `.describe()`, `.shape`, null counts, and numerical/categorical
column splits to understand the data before touching it.

**2. Missing values**
- **Age** (177 missing) → filled with the **median** (robust to skew/outliers).
- **Embarked** (2 missing) → filled with the **mode** (minimal impact).
- **Cabin** (687 missing, ~77%) → **dropped** — too sparse to impute reliably.

**3. Outliers**
Boxplot on `Fare` to flag extreme values.

**4. Visualization**
- Age distribution (histogram + KDE)
- Fare by passenger class (boxplot)
- Survival rate by sex (bar chart)
- Correlation heatmap (numerical features)

---

## Key finding

**Sex is the strongest driver of survival:** ~74% of women survived vs ~19% of men.
Passenger class is second (1st ≈ 63%, 2nd ≈ 47%, 3rd ≈ 24%), and `Pclass` has the
strongest numeric correlation with survival (-0.34; negative because a lower class
number means higher status).

---

## Stack

Python · pandas · NumPy · matplotlib · seaborn · Jupyter

---
