# Week 2: Titanic Survival Prediction - Classification

First project in the NeuroFive ML track. Takes the raw Titanic dataset through the
full early-stage ML workflow — cleaning, exploratory analysis, and a first
predictive model for passenger survival.

---

## Dataset

`titanic_train.csv` — 891 passengers, 12 columns. Target is `Survived`
(0 = did not survive, 1 = survived; ~38% survived overall).

---

## Files

| File | Contents |
|------|----------|
| `Titanic - Cleaning And Visualizing.ipynb` | Full workflow: cleaning → EDA → logistic regression model. |
| `titanic_train.csv` | Raw training dataset. |

---

## Approach

**1. Cleaning & missing values**
- **Age** (177 missing) → filled with the median (robust to skew).
- **Embarked** (2 missing) → filled with the mode.
- **Cabin** (687 missing, ~77%) → dropped; too sparse to impute reliably.

**2. Exploratory analysis**
Histograms, boxplots, a survival-by-sex bar chart, and a correlation heatmap.
Key finding: **sex is the strongest driver of survival** (~74% of women vs ~19% of
men), followed by passenger class. Age showed almost no correlation.

**3. Feature selection**
Dropped `PassengerId` (identifier), `Name` and `Ticket` (high-cardinality free
text — no useful signal, would only add noise). Kept `Pclass, Sex, Age, SibSp,
Parch, Fare, Embarked`.

**4. Encoding**
Categorical columns `Sex` and `Embarked` one-hot encoded with
`pd.get_dummies(drop_first=True)` to keep the input numeric and avoid the
dummy-variable trap.

**5. Train/test split**
80/20 split via `train_test_split` with `random_state=42` for reproducibility and
`stratify=y` to preserve the survived/died ratio across both sets (the classes are
imbalanced).

**6. Model**
`LogisticRegression(max_iter=1000)` — the raised iteration cap ensures the solver
fully converges on this data.

---

## Results

**Accuracy: 80.45%** (144 of 179 test passengers correctly classified)

Confusion matrix:

|                       | Predicted: Died | Predicted: Survived |
|-----------------------|:---------------:|:-------------------:|
| **Actually died**     | 98 (TN)         | 12 (FP)             |
| **Actually survived** | 23 (FN)         | 46 (TP)             |

The model identifies non-survivors well (~89% recall) but misses nearly a third of
actual survivors (~67% recall). This gap is a consequence of class imbalance — with
more non-survivors in the data, the model leans slightly toward predicting "died."
Accuracy alone hides this, which is why the confusion matrix is reported alongside it.

---

## Stack

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · Jupyter

---

# Week 2: California Housing Price Prediction - Linear Regression

Regression counterpart to the Titanic classification task. Takes the California
housing dataset through cleaning, EDA, feature selection, and a linear regression
model predicting median house value.

---

## Dataset

`sklearn.datasets.fetch_california_housing` — 20,640 California census block groups
from the 1990 census. Each row is a **neighbourhood**, not an individual house.

> **Note on Boston:** the Boston housing dataset was deprecated in scikit-learn 1.0
> and **removed in 1.2** over ethical concerns with one of its features. California
> housing is the intended replacement.

| Column | Meaning |
|--------|---------|
| `MedInc` | Median income in the block (units of $10,000) |
| `HouseAge` | Median house age in years |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block population |
| `AveOccup` | Average household occupancy |
| `Latitude` / `Longitude` | Geographic location |
| `MedHouseVal` | **Target** — median house value in units of $100,000 |

A target value of `2.5` means **$250,000**.

---

## Approach

**1. Exploration**
No missing values and no duplicate rows — unlike Titanic, the cleaning work here is
about outliers and censoring rather than gaps.

**2. Data quality issues found**
- **Censored target:** exactly **965 districts** sit at $500,001 because the original
  survey truncated there. These are not real prices.
- **Implausible ratios:** `AveRooms` reaches 141.9 and `AveOccup` reaches 1243 —
  values that describe institutions or resorts, not residential blocks.

**3. Cleaning**
- Dropped blocks with `AveRooms >= 20` (69 rows).
- Dropped capped target values `MedHouseVal >= 5.0` (992 rows).
- **1,060 rows removed; 19,580 retained.**

**4. Visualisation**
Target histogram (the cap is visible as a spike at the right edge), correlation
heatmap, income-vs-price scatter, and a geographic price map. The geographic plot
reproduces the shape of California with expensive coastal areas (Bay Area, LA)
standing out against the cheap interior — this is the justification for using
location as a feature.

**5. Feature selection (5 features)**

| Feature | Correlation with price | Rationale |
|---------|:----------------------:|-----------|
| `MedInc` | **0.688** | Dominant predictor — wealthier areas, pricier houses |
| `AveRooms` | 0.152 | Proxy for house size (closest stand-in for square footage) |
| `HouseAge` | 0.106 | Property age affects value |
| `Latitude` | −0.144 | Location — weak alone, strong jointly |
| `Longitude` | −0.046 | Location — weak alone, strong jointly |

Excluded: `AveBedrms` (near-duplicate of `AveRooms`), `Population` and `AveOccup`
(effectively zero correlation with price).

**6. Model**
`LinearRegression` on an 80/20 split with `random_state=42`. No feature scaling —
ordinary least squares produces identical predictions with or without it.

---

## Results

| Metric | Value |
|--------|-------|
| **RMSE** | **0.6288** (≈ **$62,875**) |
| **R²** | **0.5873** |

A predicted-vs-actual scatter plot with a perfect-prediction reference line shows
points clustering around the line but scattering widely. The model under-predicts
expensive houses and never predicts above roughly $400k — linear regression pulls
toward the mean and cannot capture the top of the market.

---

### What R² = 0.587 means ?

House prices vary a lot from neighbourhood to neighbourhood. An R² of 0.587 means
the model explains about **59% of that variation** using only income, house size,
age, and location. The remaining 41% comes from things the model cannot see —
school quality, crime rates, ocean views, property condition. The RMSE of 0.63 means
a typical prediction is off by roughly **$63,000**, so a neighbourhood estimated at
$250,000 is realistically somewhere between $190,000 and $310,000. That is useful
for spotting broad patterns but nowhere near precise enough to price an individual
house.

---

## Caveats

- **Removing the capped values improved RMSE but lowered R²** (0.605 → 0.587). Not a
  contradiction: R² is measured relative to how much the target varies, so cutting
  off the expensive tail shrinks that variance and makes explaining a fixed share of
  it harder, even as absolute error falls. RMSE is the more trustworthy metric when
  the data's range has changed.
- **`AveRooms` receives a small negative coefficient** (−0.016) despite being
  positively correlated with price on its own. This is multicollinearity — `MedInc`
  already absorbs the "bigger houses in richer areas" effect. Individual coefficients
  should not be read as causal.
- Data is from the **1990 census** and reflects prices from that period.

---

## Stack

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · Jupyter
