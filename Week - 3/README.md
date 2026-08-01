# Week 2: Titanic Survival Prediction - Classification

Takes the raw Titanic dataset through the full classification workflow — cleaning,
exploratory analysis, a baseline logistic regression model, per-class evaluation
with precision/recall/F1, and hyperparameter tuning with GridSearchCV.

---

## Dataset

`titanic_train.csv` — 891 passengers, 12 columns. Target is `Survived`
(0 = did not survive, 1 = survived). The classes are **imbalanced: 549 died vs
342 survived (~62/38)**, which drives much of the analysis below.

---

## Files

| File | Contents |
|------|----------|
| `Titanic.ipynb` | Full workflow: cleaning → EDA → logistic regression → evaluation → tuning. |
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
`stratify=y` to preserve the survived/died ratio across both sets.

**6. Baseline model**
`LogisticRegression(max_iter=1000)` — the raised iteration cap ensures the solver
fully converges on this data.

**7. Per-class evaluation**
`classification_report` to break performance down by class, exposing the gap that
accuracy conceals.

**8. Hyperparameter tuning**
`GridSearchCV` over three hyperparameters (`C`, `penalty`, `class_weight`) with
5-fold cross-validation — 24 combinations, 120 fits. Scored on **F1 rather than
accuracy**, since the goal was to fix minority-class recall. `solver='liblinear'`
was required to support the `l1` penalty, and a `StandardScaler` was bundled into a
`Pipeline` so scaling is fit inside each CV fold (no leakage).

---

## Results

### Baseline model

**Accuracy: 80.45%** (144 of 179 test passengers correctly classified)

|                       | Predicted: Died | Predicted: Survived |
|-----------------------|:---------------:|:-------------------:|
| **Actually died**     | 98 (TN)         | 12 (FP)             |
| **Actually survived** | 23 (FN)         | 46 (TP)             |

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Did not survive | 0.8099 | 0.8909 | 0.8485 | 110 |
| Survived | 0.7931 | **0.6667** | 0.7244 | 69 |

The model identifies non-survivors well but **misses a third of actual survivors**.
Precision is similar across classes; it is recall that collapses for the minority
class.

### Why accuracy alone is misleading

Accuracy averages across all classes and treats every error as equally costly, so
failures on the smaller class disappear into the average. On this test set, a model
that predicted "did not survive" for **every passenger** — learning nothing — would
still score **61.45% accuracy**. That is the floor any score must be judged against,
and it makes 80.45% look considerably less impressive.

The problem scales with imbalance: in fraud detection where 1 in 1,000 transactions
is fraudulent, a model flagging nothing is 99.9% accurate and entirely useless.
Accuracy also hides *which* error is being made — our baseline produces 23 false
negatives against only 12 false positives, and in a rescue scenario those two
mistakes carry very different costs.

### Tuned model

Best parameters: `C=10`, `penalty='l1'`, `class_weight='balanced'`
(best cross-validated F1: 0.7314)

|                       | Predicted: Died | Predicted: Survived |
|-----------------------|:---------------:|:-------------------:|
| **Actually died**     | 90 (TN)         | 20 (FP)             |
| **Actually survived** | 15 (FN)         | 54 (TP)             |

### Before / After comparison

| Metric | Before (Baseline) | After (Tuned) | Change |
|--------|:-----------------:|:-------------:|:------:|
| Accuracy | 0.8045 | 0.8045 | **0.0000** |
| Precision (Survived) | 0.7931 | 0.7297 | −0.0634 |
| Recall (Survived) | 0.6667 | **0.7826** | **+0.1159** |
| F1-Score (Survived) | 0.7244 | **0.7552** | **+0.0308** |
| Macro F1 | 0.7864 | **0.7962** | **+0.0098** |

**Accuracy did not move at all.** Judged on that metric alone, the tuning was a
complete waste of time — which is the sharpest possible illustration of the point
above. Underneath the unchanged number, recall on the Survived class rose from
0.6667 to 0.7826: **false negatives fell from 23 to 15**, so the model now finds 8
more actual survivors.

This came at a genuine cost. Precision dropped from 0.7931 to 0.7297 as false
positives rose from 12 to 20 — the classic precision/recall trade-off.
`class_weight='balanced'` makes the model take the minority class more seriously, so
it predicts "survived" more readily, catching more true survivors while also raising
more false alarms. F1 and macro F1 both rising confirms the trade was worthwhile.

Whether the tuned model is genuinely "better" depends on the cost of each error type
— a domain decision, not one any single metric can make.

---

## Notes

- Running the same grid **without** `StandardScaler` produces identical test results,
  so the improvement comes purely from the hyperparameters. The scaler is included
  as correct practice for regularized models, not because it does the work here.
- `GridSearchCV` was chosen over `RandomizedSearchCV` because 24 combinations is
  small enough to search exhaustively; randomized search only earns its keep on
  larger spaces.

---

## Stack

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · Jupyter

---

# Week 3: Telco Customer Churn - Model Comparison

Takes the IBM Telco Customer Churn dataset through cleaning, EDA, and a head-to-head
comparison of two classifiers — Decision Tree vs Logistic Regression — with feature
importance analysis and a business-facing summary.

---

## Dataset

`WA_Fn-UseC_-Telco-Customer-Churn.csv` — 7,043 customers, 21 columns. Target is
`Churn` (Yes/No). Each row is one customer, with demographics, subscribed services,
contract details, and billing information.

The classes are **imbalanced: 5,174 stayed (73.5%) vs 1,869 churned (26.5%)** — about
2.8:1. This drives much of the analysis below.

---

## Files

| File | Contents |
|------|----------|
| `Telco Churn.ipynb` | Full workflow: cleaning → EDA → both models → comparison → feature importances. |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Raw dataset. |

---

## Approach

**1. Cleaning**

`isnull()` reports zero missing values, which is misleading. `TotalCharges` is stored
as **text**, not a number — it contains 11 blank strings that pandas does not count as
null.

Critically, **all 11 blanks belong to customers with `tenure = 0`** — brand new
customers who have never been billed. Their true total charge is `0`, not "unknown",
so they were filled with 0. Dropping the rows or imputing a median would both have
been wrong.

`customerID` was dropped (unique identifier, no predictive value; zero duplicates
confirmed).

**2. Exploratory analysis**

Churn rate broken down by every categorical feature, plus distributions and
correlations for the numeric ones.

**3. Encoding**

15 categorical columns one-hot encoded with `pd.get_dummies(drop_first=True)` to avoid
the dummy-variable trap. Shape goes from `(7043, 20)` to `(7043, 31)`. `SeniorCitizen`
needed no encoding — already stored as 0/1.

**4. Split**

80/20 via `train_test_split` with `random_state=42` and `stratify=y` to preserve the
73/27 ratio across both sets.

**5. Models**

- **Decision Tree** — `max_depth=5`, `min_samples_leaf=50`
- **Logistic Regression** — wrapped in a `Pipeline` with `StandardScaler`, since
  `TotalCharges` reaches ~8,700 while dummy columns are 0/1. The pipeline keeps
  scaling inside the training fold, preventing leakage.

---

## EDA Findings

| Feature | Churn rate |
|---------|:----------:|
| **Contract: Month-to-month** | **42.7%** |
| Contract: One year | 11.3% |
| **Contract: Two year** | **2.8%** |
| **Payment: Electronic check** | **45.3%** |
| Payment: Credit card (automatic) | 15.2% |
| **Internet: Fiber optic** | **41.9%** |
| Internet: DSL | 19.0% |
| TechSupport: No | 41.6% |
| SeniorCitizen: Yes | 41.7% |
| gender: Female / Male | 26.9% / 26.2% |

**Contract type is the clearest signal** — a 15× gap between month-to-month and
two-year customers.

**Tenure is the strongest numeric predictor** (r = −0.352). Churners average 18.0
months with the company versus 37.6 for those who stay; churn risk is front-loaded.

**Fiber optic churns at more than double the DSL rate**, despite being the premium
product — counterintuitive and worth investigating as either price sensitivity or a
service quality problem.

**Gender has essentially no effect** and contributes nothing to either model.

---

## Results

### Overfitting check

An unconstrained Decision Tree reached **depth 23 with 1,102 leaves**, scoring 99.8%
on training data but only **72.6% on test** — worse than predicting "no churn" for
everyone. Depth and leaf-size limits were applied in response.

### Model comparison

| Metric | Decision Tree | Logistic Regression | Winner |
|--------|:-------------:|:-------------------:|:------:|
| Accuracy | 0.7928 | **0.8070** | LR |
| Precision (Churn) | 0.6367 | **0.6584** | LR |
| Recall (Churn) | 0.5107 | **0.5668** | LR |
| F1 (Churn) | 0.5668 | **0.6092** | LR |
| ROC-AUC | 0.8233 | **0.8418** | LR |

**Logistic Regression wins on every metric.** The relationships here are largely
linear and additive, which suits LR; a single tree must carve the space into
rectangles and loses information doing so.

Context matters: the **majority-class baseline is 73.46% accuracy**, so LR's 80.70% is
only ~7 points above predicting that nobody churns.

### Class imbalance

Both models show the classic symptom — recall on the minority Churn class is weak
(0.51 and 0.57), meaning **roughly half of actual churners are never flagged**. This
is the business-critical error, since an undetected churner is a lost customer.

`class_weight='balanced'` was applied as a **partial** fix:

| Metric | DT | DT balanced | LR | LR balanced |
|--------|:--:|:-----------:|:--:|:-----------:|
| Accuracy | 0.7928 | 0.7452 | 0.8070 | 0.7402 |
| Recall (Churn) | 0.5107 | **0.8048** | 0.5668 | **0.7861** |
| F1 (Churn) | 0.5668 | **0.6264** | 0.6092 | **0.6164** |

Churn recall rises from ~51% to **~80%** while accuracy falls. For a retention use
case that is the correct trade — accuracy is the wrong objective here. SMOTE
oversampling or decision-threshold tuning would be the fuller treatment.

### Top 3 churn drivers (`.feature_importances_`)

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | **tenure** | 0.4338 |
| 2 | **InternetService_Fiber optic** | 0.3633 |
| 3 | **PaymentMethod_Electronic check** | 0.0373 |

Verified stable across `max_depth` 3–6, so not an artifact of one specific tree.

---

## Business Summary

We can now predict which customers are likely to leave with about **81% accuracy**,
and more usefully, we have identified *why* they leave. **The first year is the danger
zone** — customers who leave do so after an average of 18 months, versus 38 months for
those who stay, so early-relationship retention is where effort pays off most. **Three
specific groups are at high risk:** month-to-month customers (43% leave, versus just
3% on two-year contracts), fiber optic internet customers (42% leave, more than double
the DSL rate), and customers paying by electronic check (45% leave, roughly triple
those on automatic payment).

The recommended actions follow directly: offer targeted incentives to move
month-to-month customers onto annual contracts, investigate whether the fiber optic
churn problem is about pricing or service quality since it is our premium product
losing customers fastest, and promote automatic payment enrolment. One caution — the
model currently catches only about half of the customers who will actually leave, so
it should guide where to focus retention spending rather than serve as a definitive
list. Tuning it to flag more at-risk customers is a straightforward next step.

---

## Caveats

- **`Contract` barely registers in the tree's feature importances** despite being the
  strongest EDA signal. This is collinearity, not contradiction: month-to-month
  customers average 18 months tenure vs 56.7 for two-year customers (r = 0.56), so
  splitting on `tenure` first absorbs most of the contract signal.
  `feature_importances_` measures *what the tree used*, not what matters in the
  world. Logistic Regression coefficients confirm Contract's real effect —
  `Contract_Two year` carries one of the strongest retention weights.
- **A single Decision Tree is a weak model here.** Random Forest or Gradient Boosting
  would likely beat both models and would give more reliable importances, since
  averaging across many trees reduces the collinearity artifact above.

---

## Stack

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · Jupyter
