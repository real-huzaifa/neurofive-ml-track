# 🎮 Steam Game Reception Predictor

Predicting how a game will be received on Steam — using only the decisions a developer makes
**before** launch.

**[▶ Try the live app](https://steam-game-reception-predictor.streamlit.app/)**

---

## Overview

Every year tens of thousands of games launch on Steam, and most disappear. A solo developer or
small studio makes a series of consequential decisions months before release — what to price
it at, which genres to target, how many languages to localise into, which month to ship in —
and then finds out whether those decisions worked only after it is too late to change them.

This project builds a tool that estimates a game's likely reception from exactly those
pre-launch decisions, trained on **125,855 games** scraped from the Steam API.

It is deliberately not a "will my game be good?" predictor. It answers a narrower, more
honest question: *how have games with this profile historically been received?*

---

## Problem Statement

**Predict the percentage of positive reviews a Steam game will receive, using only features
knowable before it ships.**

The constraint in that sentence is the entire project.

Most public analyses of Steam data predict review scores using features like review counts,
owner estimates, peak concurrent players, or playtime. Those are all **post-launch outcomes**.
A model built on them scores well and is useless — by the time you know your review count, the
launch already happened and the model has nothing left to inform.

So every feature here had to pass one test: *would a developer know this the day before
release?*

| Used as features (pre-launch) | Deliberately excluded (post-launch) |
|---|---|
| Price | Review counts |
| Genres, Steam categories | Estimated owners |
| Supported languages | Peak concurrent players |
| Platforms (Windows / macOS / Linux) | Average & median playtime |
| Achievement count | Metacritic / user scores |
| Required age rating | Recommendations |
| Release month and year | Current discount |
| Developer's prior release count | |

The developer track-record feature needed care of its own. A naïve "how many games has this
studio made?" would count *future* releases too. It is built by sorting the whole dataset by
release date and taking a cumulative count, so for any given game it only ever counts titles
that already existed.

---

## Dataset

**Source:** [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
(Kaggle, built from the official Steam Web API) — 125,855 games, 40 columns.

### A header bug that silently corrupts the file

The CSV ships with **39 header names but 40 data columns**. The eighth header,
`DiscountDLC count`, is two columns (`Discount` and `DLC count`) merged into a single name.

Load it with `pd.read_csv()` normally and every column from index 7 onward shifts by one —
prices land in the discount column, genres land in the languages column, and nothing errors.
The fix is to supply corrected column names explicitly:

```python
CORRECTED_COLUMNS = [..., 'Price', 'Discount', 'DLC count', 'About the game', ...]
df = pd.read_csv('games.csv', header=0, names=CORRECTED_COLUMNS, usecols=KEEP)
```

### Eligibility filter

Only games with **at least 50 reviews** were kept — 30,541 of 125,855 (24.3%).

Below that threshold the target is noise. A game with three reviews at 100% positive is not
meaningfully better received than one with five hundred at 95%; it just has fewer data points.
Fifty is a defensible floor for a percentage to mean something.

---

## Approach

```
Raw CSV (125,855 games)
   ↓  fix malformed header
   ↓  filter to ≥50 reviews, valid date, valid genres  →  30,541 games
   ↓  engineer pre-launch features only                →  32 features
   ↓  train regression + classification models
   ↓  validate on a random split AND forward in time
   ↓  deploy both models as a Streamlit app
```

### Two models, two framings

| Model | Task | Output |
|---|---|---|
| **Regressor** | Predict % positive reviews | "82.4% positive — *Very Positive* band" |
| **Classifier** | Clears Steam's 70% "Positive" bar? | "Likely well-received — 83% confidence" |

The app runs both and reports them side by side, including when they disagree.

---

## Results

### Regression — predicting % positive

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (predict the mean) | 15.459 | 12.243 | −0.000 |
| Linear Regression | 14.249 | 11.133 | 0.150 |
| Random Forest | 13.495 | 10.524 | 0.238 |
| **HistGradientBoosting** | **13.449** | **10.422** | **0.249** |

### Classification — clearing the 70% bar

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (always predict "yes") | 0.7518 | 0.7518 | 1.0000 | 0.8583 | 0.5000 |
| Logistic Regression | 0.7518 | 0.7682 | 0.9595 | 0.8532 | 0.6950 |
| Random Forest | 0.7757 | 0.7943 | 0.9469 | 0.8639 | 0.7450 |
| **HistGradientBoosting** | 0.7749 | 0.7933 | 0.9475 | 0.8636 | **0.7472** |
| HistGB (`class_weight='balanced'`) | 0.7013 | 0.8634 | 0.7159 | 0.7828 | 0.7490 |

### Does a dedicated classifier beat thresholding the regressor?

A regressor spends capacity distinguishing 20% from 30%, which nobody cares about. A
classifier concentrates entirely on the boundary at 70%. So it should be sharper — but is it?

| Approach | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Thresholded regressor | **0.7762** | **0.8074** | 0.9225 | 0.8611 |
| Dedicated classifier | 0.7749 | 0.7933 | **0.9475** | **0.8636** |

**Essentially tied.** Neither dominates — the thresholded regressor edges precision, the
classifier edges recall. The practical conclusion is that a single regression model can serve
both outputs with negligible loss.

### Forward-in-time validation

A random split lets a model train on 2024 games and test on 2019 ones. A real tool has to
predict forward, so the model was retrained on games released up to 2022 and tested on 2023+.

| Split | RMSE | R² |
|---|---|---|
| Random 80/20 | 13.449 | 0.249 |
| **Time-based (≤2022 → 2023+)** | **12.342** | **0.147** |

R² drops from 0.249 to 0.147. The cause is visible in the means: pre-2023 games averaged
**77.1%** positive while 2023+ games averaged **83.0%**. There is genuine distribution shift
over time, and a model trained on the past systematically under-predicts newer releases.

---

## Key Findings

### 1. Metadata explains about a quarter of reception — and that is the honest ceiling

R² of 0.249 is modest, and it should be. Store-page metadata cannot capture whether a game is
actually *good*, which is what most of the remaining variance is. A model claiming R² of 0.8
here would be leaking post-launch data.

### 2. Both accuracy *and* F1 misled on the classification task

75% of eligible games clear the 70% bar, so predicting "yes" for everything scores **0.7518
accuracy and 0.8583 F1**.

Logistic Regression matched that accuracy exactly and scored *worse* on F1 (0.8532). By those
two metrics it looks like a failure. But its ROC-AUC was **0.695 against a 0.500 coin flip** —
it had genuinely learned to rank games, it simply could not beat a trivial rule at the hard
yes/no call.

The usual lesson is "accuracy misleads on imbalanced data." Here F1 misled too. ROC-AUC was
the only metric that revealed real signal.

### 3. The language paradox — why this tool reports, and does not advise

Holding everything else fixed, the model predicts:

| Languages supported | Predicted reception |
|---|---|
| 1 | 88.1% |
| 5 | 82.8% |
| 10 | 80.2% |

**More localisation predicts worse reception.** This is almost certainly not causal. Games
localised into many languages tend to be larger commercial projects carrying higher player
expectations, while English-only indies reach smaller, more sympathetic audiences. The model
has learned an association between localisation and project scale.

If this app presented itself as an optimiser — "adjust these sliders to improve your score" —
it would advise developers to **drop their translations**, which is absurd. That single finding
shaped the product: the app reports *"games with this profile historically scored around X"*
and carries an explicit warning against causal interpretation.

### 4. Release year dominates feature importance, and that is a caveat rather than an insight

| Feature | Permutation importance |
|---|---|
| release_year | 0.2112 |
| Price | 0.0547 |
| Steam Cloud integration | 0.0317 |
| Number of Steam categories | 0.0314 |
| Multi-player support | 0.0295 |
| Languages supported | 0.0271 |
| Platform count | 0.0251 |

Release year being the strongest feature reflects the temporal drift found in the time-based
validation, not anything about game design. Among features a developer actually controls,
**price** leads, followed by breadth signals — Steam features integrated, languages, platforms
— which plausibly proxy for development investment. Stated as a hypothesis, not a conclusion.

Permutation importance was used rather than the built-in impurity measure, because it reports
what actually happens to accuracy when a feature is shuffled and is harder to fool.

---

## The App

Built with Streamlit, serving both models from a single joblib bundle.

- Configure a hypothetical game: price, genres, Steam features, languages, platforms,
  achievements, age rating, release timing, studio track record
- Get the regressor's predicted percentage with its Steam band, alongside the classifier's
  verdict and confidence
- **Explicit disagreement handling** — when the two models split, the app says so and flags the
  game as sitting near the decision boundary, rather than hiding the conflict
- A prominent caveat explaining the association-not-causation limitation

Two engineering details worth noting:

**Feature order is enforced.** The app rebuilds a `pandas.DataFrame` reindexed to the model's
exact saved column order. With a DataFrame, a wrong order raises a `ValueError`; with a raw
NumPy array it silently predicts nonsense.

**Release year is clamped to the training range.** Tree models cannot extrapolate beyond the
values they were trained on, so a user selecting 2027 is handled explicitly and told what
happened, rather than being given a quietly meaningless number.

---

## Tools

| Area | Stack |
|---|---|
| Data handling | pandas, NumPy |
| Modelling | scikit-learn (`HistGradientBoosting`, `RandomForest`, `LinearRegression`, `LogisticRegression`) |
| Interpretation | scikit-learn `permutation_importance` |
| Visualisation | Matplotlib, Seaborn |
| Serialisation | joblib |
| App & deployment | Streamlit, Streamlit Community Cloud |
| Environment | Google Colab |

**Why scikit-learn's `HistGradientBoosting` rather than XGBoost:** the deployed model must
install inside Streamlit Community Cloud's free tier, and XGBoost's ~58 MB package exceeds the
available memory. `HistGradientBoosting` is the same histogram-based gradient boosting
algorithm, ships inside a library the app already needs, and performed comparably here.

---

## Limitations

- **Correlational, not causal.** Every finding describes association in historical data. No
  input change reliably causes a change in outcome.
- **Survivorship in the eligibility filter.** Requiring 50+ reviews excludes 76% of Steam's
  catalogue — largely games that got no traction at all. Conclusions apply to games that
  achieved at least modest visibility, not to the full catalogue.
- **Temporal drift.** Reception scores have risen over time. Forward-in-time R² of 0.147 is
  the more realistic expectation for predicting a future release than the 0.249 from a random
  split.
- **No content signal.** Nothing here captures whether a game is fun, polished, or bug-free —
  which is most of what determines reception.
- **Steam tags excluded.** Tags are partly community-assigned after launch, making them an
  ambiguous pre-launch signal. They were left out of the primary model rather than quietly
  included.
