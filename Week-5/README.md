## Week - 5: Handling Imbalanced And Messy Real World Data

---

**Dataset:** Credit Card Fraud (284,807 transactions, 31 columns)

Detecting fraud in a severely imbalanced dataset, and demonstrating why accuracy is the wrong
metric for the job.

---

### Class balance

After removing 1,081 duplicate rows: **283,726 transactions, 473 fraudulent — 0.167%**, a
**599:1 imbalance**.

Duplicates were dropped first because, with so few frauds, identical rows spanning the
train/test split would leak and inflate scores.

The distribution is plotted on both a linear and a log scale — on a linear axis the fraud bar
is effectively invisible, which is itself a useful illustration of the problem.

---

### Techniques applied

Three approaches to the imbalance, all compared against an untouched baseline:

| Technique | How it works |
|---|---|
| `class_weight='balanced'` | Penalises errors on the minority class more heavily during training |
| **SMOTE** | Generates synthetic minority examples by interpolating between real ones |
| **Random undersampling** | Discards majority rows until the classes are even |

Resampling was applied to the **training set only**. Running SMOTE before the split would place
synthetic frauds — interpolated from real ones — into the test set, making the scores fiction.

---

### Results

80/20 stratified split · 95 frauds in the test set · logistic regression throughout

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Predict "never fraud" | 0.9983 | 0.000 | 0.000 | 0.000 | 0.002 | 0 | 0 | 95 |
| Baseline (no balancing) | 0.9991 | **0.846** | 0.579 | **0.688** | **0.692** | 55 | 10 | 40 |
| `class_weight='balanced'` | 0.9753 | 0.056 | **0.874** | 0.106 | 0.672 | 83 | 1389 | 12 |
| SMOTE | 0.9737 | 0.053 | **0.874** | 0.100 | 0.675 | 83 | 1482 | 12 |
| Random undersampling | 0.9722 | 0.050 | **0.874** | 0.095 | 0.590 | 83 | 1563 | 12 |

---

### Before and after

All three techniques did exactly what they are designed to do: **recall rose from 0.579 to
0.874** — the model now catches 83 of 95 frauds instead of 55, missing 12 instead of 40.

But **precision collapsed from 0.846 to roughly 0.05**, and F1 fell from 0.688 to about 0.10.
False positives went from 10 to over 1,400.

---

### Why F1 dropped — the threshold, not the model

PR-AUC is nearly identical between the baseline (0.692) and the balanced model (0.672). Since
PR-AUC is threshold-free, the two models rank transactions almost equally well. What balancing
changed is **where the default 0.5 cutoff falls** relative to those scores.

Tuning the threshold on the balanced model instead:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.5 (default) | 0.056 | 0.874 | 0.106 |
| 0.9 | 0.263 | 0.832 | 0.399 |
| 0.99 | 0.658 | 0.790 | 0.718 |
| 0.999 | 0.745 | 0.768 | **0.757** |

At its optimum the balanced model reaches **F1 = 0.814**, against **0.791** for the baseline. So
balancing did help — but only once the threshold moved with it.

**Class balancing and threshold tuning solve the same problem two different ways.** Applying
balancing and then leaving the cutoff at 0.5 is what produces the 1,400-false-positive result.

Which model to deploy is a cost question: a missed fraud costs the transaction amount plus
reputational damage, while a false positive costs an analyst's review time or an annoyed
customer. At 1,400 false positives per 56,746 transactions a review team would drown; the
baseline's 10 is operationally sane but misses 40 frauds. A real deployment would tune the
threshold to the team's actual review capacity — which the table above makes possible.

---

### Why "accuracy" would have been a misleading metric

Only 0.167% of transactions in this dataset are fraudulent — 473 out of 283,726. A model that
simply predicts "not fraud" for every single transaction achieves **99.83% accuracy** while
catching **zero** frauds. I verified this directly with a `DummyClassifier`: accuracy 0.9983,
recall 0.000, F1 0.000. By the accuracy metric it looks near-perfect; in reality it is
completely worthless, because the entire purpose of the model is to find the 0.167% it ignores.

The problem is that accuracy weights every transaction equally, so on imbalanced data it mostly
measures how large the majority class is. Worse, it points the wrong way here: the baseline
scored **99.91% accuracy** while the balanced model scored **97.53%** — yet the balanced model
caught 83 frauds to the baseline's 55. Optimising for accuracy would have led me to choose the
model that misses more fraud.

The right metrics focus on the minority class: **recall** (what fraction of real frauds did we
catch?), **precision** (of the transactions we flagged, how many were actually fraud?), **F1**
(their harmonic mean), and **PR-AUC**, which summarises the precision–recall trade-off across
all thresholds without depending on any single cutoff. For a fraud system the choice between
them is a business decision — the cost of a missed fraud versus the cost of investigating a
false alarm — and accuracy answers neither question.

---

## Week - 5: Deploy Your Model As A Live Web App

Interactive Streamlit app serving a trained gradient boosting model. Pick a California city,
adjust the block group's income, age, and household details, and get a predicted median house
value.

---

## 🚀 Live Demo

**[Live app →](https://calfornia-housing-prediction.streamlit.app/)**

Taking the best-performing model from an earlier task and serving it as an interactive web app.

---

### The model

From an earlier ensemble comparison on the California Housing dataset (20,640 census block
groups, 8 features, target = median house value in $100,000s):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.7456 | 0.5332 | 0.5758 |
| Random Forest | 0.5053 | 0.3275 | 0.8051 |
| **XGBoost** | **0.4626** | **0.3107** | **0.8367** |

XGBoost was the best performer and the natural candidate to deploy.

---

### What the app does

- Select a California city (which sets latitude/longitude), or enter coordinates manually
- Adjust median income, house age, average rooms, bedrooms, occupancy, and population
- Click **Predict** for an estimated median house value
- Warns the user when a prediction hits the dataset's $500,001 ceiling

---

### Deployment stack

Model saved with `joblib` → Streamlit app → deployed free on Streamlit Community Cloud.

The model is bundled with its feature names and metrics rather than saved bare:

```python
joblib.dump({
    "model": final_model,
    "feature_names": list(X.columns),
    "metrics": {"RMSE": ..., "MAE": ..., "R2": ...},
}, "housing_model.joblib")
```

---

### Three problems solved during deployment

**XGBoost didn't fit the free tier.** Its package is ~58 MB and the Streamlit Community Cloud
build hung during install. The deployed model is scikit-learn's
**`HistGradientBoostingRegressor`** — the same histogram-based gradient boosting algorithm,
built into a library the app already required, performing comparably on this data. This is a
hosting decision, not a change to the comparison results above.

**Feature order is a silent failure mode.** The app rebuilds a `pd.DataFrame` with columns in
the model's exact saved order before predicting. With a DataFrame, a wrong column order raises
a `ValueError`; with a raw NumPy array it silently returns a wrong number.

**scikit-learn must be pinned to the exact training version.** Pickle stores references to
internal module paths that scikit-learn moves between releases. A mismatch produces
`ModuleNotFoundError: No module named '_loss'` at load time. The version pin is part of the
artifact, not an optimisation.

---

### Data limitation surfaced in the app

The original 1990 survey capped median house value at $500,001, so 992 rows (4.8%) sit at that
ceiling and the model cannot predict above it. Rather than showing a falsely precise figure
like "$499,847", the app tells the user the prediction is a floor.

---

## Tools

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · Matplotlib · joblib ·
Streamlit
