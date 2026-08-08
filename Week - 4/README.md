## Week - 4: Build a Proper ML Pipeline with Feature Engineering
 
Building a single reproducible `Pipeline` that handles all preprocessing and modelling in one
object, then testing whether engineered features improve it.

---

### Dataset
 
Titanic training set — 891 passengers, 12 columns, binary target (`Survived`: 342 survived /
549 did not).
 
Missing values: `Age` 177 · `Cabin` 687 · `Embarked` 2

---

### Approach
 
A `ColumnTransformer` routes each column type down its own branch, and the whole thing is
chained to the model:
 
```
Pipeline
├── ColumnTransformer
│   ├── numeric      → SimpleImputer(median)        → StandardScaler
│   ├── categorical  → SimpleImputer(most_frequent) → OneHotEncoder
│   └── passthrough  → (already 0/1 flags)
└── LogisticRegression
```
 
Imputation sits **inside** the pipeline for two reasons: `StandardScaler` raises an error on
NaN, and fitting the imputer inside each CV fold prevents test-fold statistics leaking into
training.

---

### Engineered features
 
| Feature | Definition |
|---|---|
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `FamilySize == 1` |
| `Title` | Extracted from `Name`, grouped to Mr / Miss / Mrs / Master / Rare |
| `HasCabin` | Whether a cabin was recorded at all |

---

### Results
 
5-fold stratified cross-validation:
 
| Approach | Accuracy | ROC-AUC |
|---|---|---|
| Manual preprocessing | 0.7969 | 0.8511 |
| Pipeline (same features) | 0.8036 | 0.8511 |
| **Pipeline + engineered features** | **0.8294** | **0.8718** |
 
Held-out 20% test set: accuracy **0.8156**, ROC-AUC **0.8676**.

---

### Ablation — which features actually helped
 
| Configuration | Accuracy | ROC-AUC |
|---|---|---|
| Baseline (no engineering) | 0.8036 | 0.8511 |
| + FamilySize & IsAlone | 0.8014 | 0.8558 |
| + Title only | 0.8227 | 0.8694 |
| + HasCabin only | 0.8036 | 0.8554 |
| **All engineered** | **0.8294** | **0.8718** |
 
**`Title` carries almost the entire gain.** FamilySize/IsAlone slightly *reduce* accuracy on
their own, and HasCabin moves it not at all — only in combination does the full set reach
0.8294. Title works because it encodes age, sex, and social status simultaneously ("Master" =
boy, "Mrs" = married woman) and is present for all 891 rows, unlike `Age`.

---

### Notes
 
- The manual-vs-pipeline accuracy gap traces to one detail: the manual version scaled the
  one-hot dummy columns too, which changes how L2 regularisation penalises them. Scaling only
  the true numerics brings manual to 0.8025 — effectively identical.
- Title extraction has a trap: the raw data contains `"the Countess"`, not `"Countess"`, so an
  explicit rare-title list silently misses it. Keeping the four common titles and mapping
  everything else to `Rare` is robust.
- Final pipeline saved with `joblib`; reload verified to produce identical predictions, and it
  accepts raw input containing NaN without external preprocessing.

---
 
## Week - 4: Ensemble Learning: Random Forest vs XGBoost
 
Comparing a linear baseline against two ensemble methods on a regression task, and examining
what each model considers important.

---

### Dataset
 
California Housing from `sklearn.datasets` — 20,640 block groups, 8 features, target is median
house value in $100,000s.
 
No missing values, no duplicates. Two data-quality notes documented below.

---

## Results

Predicting median house value (in $100,000s) from 8 census-derived features.
20,640 samples · 80/20 train-test split · random_state=42

| Model | RMSE | MAE | R² | Train time |
|---|---|---|---|---|
| Linear Regression (baseline) | 0.8517 | 0.5542 | 0.4464 | <0.1s |
| Random Forest Regressor | 0.5068 | 0.3299 | 0.8040 | ~12s |
| **XGBoost Regressor** | **0.4651** | **0.3139** | **0.8349** | ~0.3s |

**Best model:** XGBoost — 45.4% lower RMSE than the linear baseline, and R² of 0.835 vs 0.446.

**Key findings**
- Both ensembles vastly outperform linear regression, indicating strongly non-linear
  relationships that a straight line cannot capture.
- XGBoost beat Random Forest while training roughly 40× faster.
- Median income is the dominant predictor for both models (~50% of total importance).
- The two models' importance rankings agree closely (Spearman ρ = 0.95); XGBoost weights
  longitude slightly higher, RF weights median income slightly higher.

**Data note:** 992 rows (4.8%) sit at the target ceiling of 5.0 ($500,001 cap in the original
survey), which places a hard floor on achievable error. 112 rows contain implausible
per-household ratios (e.g. AveOccup up to 1,243) and were retained so all models saw identical
data.

---

## Feature importances
 
| Feature | Random Forest | XGBoost | RF rank | XGB rank |
|---|---|---|---|---|
| MedInc | 0.5249 | 0.4799 | 1 | 1 |
| AveOccup | 0.1384 | 0.1586 | 2 | 2 |
| Latitude | 0.0889 | 0.0962 | 3 | 4 |
| Longitude | 0.0886 | 0.1088 | 4 | 3 |
| HouseAge | 0.0546 | 0.0656 | 5 | 5 |
| AveRooms | 0.0443 | 0.0426 | 6 | 6 |
| Population | 0.0306 | 0.0233 | 7 | 8 |
| AveBedrms | 0.0296 | 0.0251 | 8 | 7 |
 
The two models largely agree — Spearman rank correlation **0.95**. Both place median income far
ahead of everything else (~50% of total importance). Only two pairs swap, and both are
near-ties: Latitude/Longitude differ by 0.0003 in RF, and Population/AveBedrms sit at the
bottom either way.
 
The one substantive difference: XGBoost leans harder on `Longitude` (0.109 vs 0.089), spreading
more weight onto geography, while Random Forest concentrates more on `MedInc`. That fits how
boosting works — later trees chase the residuals earlier trees missed, and location is exactly
the kind of interaction effect that surfaces in what income alone cannot explain.

---

### How Random Forest and XGBoost differ in combining models
 
Random Forest uses **bagging**: it trains many deep decision trees *in parallel*, each on a
random bootstrap sample of rows and a random subset of features at each split, then averages
their predictions. Because the trees are built independently and never see each other's
mistakes, averaging mainly reduces **variance** — it smooths out noise that any single deep
tree would overfit to.
 
XGBoost uses **boosting**: it builds shallow trees *sequentially*, each one trained
specifically to correct the errors left by those before it, with every tree's contribution
shrunk by a learning rate. Because each round targets the current residuals, boosting reduces
**bias** as well as variance, which is why it usually edges out a forest on structured data —
but it is more sensitive to overfitting and needs its learning rate and tree count tuned with
more care.

---

### Data-quality notes
 
- **992 rows (4.8%) sit at the target ceiling of 5.0.** The original survey capped median house
  value at $500,001, flattening every expensive block group to the same number. No model can
  predict above that ceiling, so part of the remaining error is a data artifact rather than a
  modelling failure.
- **112 rows contain implausible per-household ratios** — `AveOccup` reaches 1,243 and
  `AveRooms` reaches 141, almost certainly block groups with near-zero households creating
  division artifacts. These were retained so all three models saw identical data; removing them
  would be a follow-up experiment, measured rather than assumed.

---
 
## Tools
 
Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · joblib
