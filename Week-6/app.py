"""
Steam Game Reception Predictor
Predicts how a game will be received on Steam from pre-launch decisions only.
Serves two scikit-learn models: a regressor (% positive) and a classifier (pass/risk).
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Steam Reception Predictor", page_icon="🎮", layout="wide")

MODEL_PATH = Path(__file__).parent / "steam_model.joblib"


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


try:
    B = load_bundle()
except FileNotFoundError:
    st.error(f"Model file not found at `{MODEL_PATH}`. "
             "Make sure `steam_model.joblib` sits next to `app.py`.")
    st.stop()

REGRESSOR = B["regressor"]
CLASSIFIER = B["classifier"]
FEATURES = B["feature_names"]
GENRES = B["top_genres"]
CATS = B["top_cats"]
M = B["metrics"]
YEAR_MAX = B["train_year_max"]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# ---------------------------------------------------------------- header
st.title("🎮 Steam Game Reception Predictor")
st.markdown(
    "Estimates how a game will be received on Steam — using **only decisions a developer "
    "makes before launch**. No review counts, no player numbers, no post-launch data."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games trained on", f"{B['n_train']:,}")
c2.metric("Regression R²", f"{M['reg']['R2']:.3f}")
c3.metric("Typical error", f"±{M['reg']['MAE']:.1f} pts")
c4.metric("Classifier ROC-AUC", f"{M['cls']['roc_auc']:.3f}")

st.divider()

# ---------------------------------------------------------------- inputs
st.subheader("Describe your game")

left, mid, right = st.columns(3)

with left:
    st.markdown("**Pricing & timing**")
    price = st.slider("Price (USD)", 0.0, 60.0, 9.99, 0.5,
                      help="Set to 0.00 for free-to-play.")
    month = st.selectbox("Release month", MONTHS, index=9)
    year = st.number_input("Release year", 2015, YEAR_MAX + 2, YEAR_MAX, 1)
    prior = st.slider("Games you've released before", 0, 30, 0,
                      help="Your studio's track record on Steam.")

with mid:
    st.markdown("**Genres**")
    picked_genres = st.multiselect("Genres", GENRES, default=["Indie", "Action"])
    st.markdown("**Reach**")
    n_lang = st.slider("Languages supported", 1, 30, 3)
    win = st.checkbox("Windows", value=True)
    mac = st.checkbox("macOS", value=False)
    linux = st.checkbox("Linux", value=False)

with right:
    st.markdown("**Steam features**")
    picked_cats = st.multiselect("Features integrated", CATS,
                                 default=["Single-player", "Steam Achievements"])
    achievements = st.slider("Number of achievements", 0, 200, 25)
    age = st.selectbox("Required age rating", [0, 12, 16, 18], index=0)

# ---------------------------------------------------------------- predict
st.divider()

if st.button("Predict reception", type="primary", use_container_width=True):

    if not picked_genres:
        st.warning("Pick at least one genre.")
        st.stop()

    n_platforms = int(win) + int(mac) + int(linux)
    if n_platforms == 0:
        st.warning("Select at least one platform.")
        st.stop()

    # Build the row with every feature the model expects, defaulted to 0
    row = {f: 0 for f in FEATURES}
    row.update({
        "Price": price,
        "Required age": age,
        "Achievements": achievements,
        "n_genres": len(picked_genres),
        "n_categories": len(picked_cats),
        "n_languages": n_lang,
        "n_platforms": n_platforms,
        "is_free": int(price == 0),
        "has_achievements": int(achievements > 0),
        "dev_prior_games": prior,
        # The model never saw years beyond its training range, so clamp rather
        # than ask a tree model to extrapolate into territory it cannot reach.
        "release_year": min(year, YEAR_MAX),
        "release_month": MONTHS.index(month) + 1,
    })
    for g in picked_genres:
        row["genre_" + g.replace(" ", "_")] = 1
    for c in picked_cats:
        row["cat_" + c.replace(" ", "_").replace("-", "_")] = 1

    # Reindex to the model's exact saved column order.
    # With a DataFrame a wrong order raises; with a raw array it silently misfires.
    X = pd.DataFrame([row])[FEATURES]

    pct = float(REGRESSOR.predict(X)[0])
    verdict = int(CLASSIFIER.predict(X)[0])
    prob = float(CLASSIFIER.predict_proba(X)[0][1])

    a, b = st.columns(2)

    with a:
        st.markdown("### Model 1 — Regression")
        st.metric("Predicted positive reviews", f"{pct:.1f}%")
        if pct >= 95:   band = "Overwhelmingly Positive"
        elif pct >= 80: band = "Very Positive"
        elif pct >= 70: band = "Positive"
        elif pct >= 40: band = "Mixed"
        else:           band = "Mostly Negative"
        st.caption(f"Steam band: **{band}** · typical error ±{M['reg']['MAE']:.1f} points")

    with b:
        st.markdown("### Model 2 — Classifier")
        if verdict:
            st.metric("Verdict", "Likely well-received")
            st.caption(f"Confidence: **{prob:.0%}** · trained on the ≥70% threshold")
        else:
            st.metric("Verdict", "At risk")
            st.caption(f"Confidence it clears 70%: **{prob:.0%}**")

    # The two models are independent — showing disagreement is more honest than hiding it
    if (pct >= 70) != bool(verdict):
        st.warning(
            f"**The two models disagree.** The regressor predicts {pct:.1f}% "
            f"(which would {'clear' if pct >= 70 else 'miss'} the 70% bar), while the "
            f"classifier says {'it clears' if verdict else 'it misses'} it. "
            "This game sits near the decision boundary — treat the result as genuinely uncertain."
        )
    else:
        st.info("Both models agree.")

    if year > YEAR_MAX:
        st.caption(f"Note: training data ends at {YEAR_MAX}, so release year was "
                   f"treated as {YEAR_MAX}.")

    with st.expander("See the values sent to the models"):
        shown = X.T.rename(columns={0: "value"})
        st.dataframe(shown[shown["value"] != 0], use_container_width=True)

# ---------------------------------------------------------------- caveats
st.divider()

st.error(
    "**Read this before acting on a prediction.** These models learned *associations* in "
    "historical Steam data — not causes. Changing an input does **not** reliably change your "
    "game's fate.\n\n"
    "The clearest example: the model predicts games supporting **more languages** score "
    "*worse*. Localised games tend to be larger commercial projects carrying higher player "
    "expectations, while English-only indies reach smaller, more sympathetic audiences. "
    "Dropping your translations would not improve your reviews.\n\n"
    "Read the output as *\"games with this profile historically scored around X%\"*, "
    "never as *\"do this and you will score X%\"*."
)

with st.expander("About these models"):
    st.markdown(
        f"""
Two scikit-learn `HistGradientBoosting` models trained on **{B['n_train']:,} Steam games**
with at least 50 reviews, released {B['train_year_min']}–{B['train_year_max']}.

| Model | Task | Performance (held-out 20%) |
|---|---|---|
| Regressor | Predict % positive reviews | RMSE {M['reg']['RMSE']:.2f} · MAE {M['reg']['MAE']:.2f} · R² {M['reg']['R2']:.3f} |
| Classifier | Clears the 70% bar? | Accuracy {M['cls']['accuracy']:.3f} · F1 {M['cls']['f1']:.3f} · ROC-AUC {M['cls']['roc_auc']:.3f} |

**Every feature is pre-launch knowable.** Review counts, owner estimates, player counts,
playtimes and Metacritic scores were all deliberately excluded — a tool that needs your
review count to predict your review score would be useless.

**R² of {M['reg']['R2']:.2f} is the honest ceiling here.** Store-page metadata explains
roughly a quarter of the variation in reception. The rest is whether the game is actually
good — which no column in this dataset captures.

**Accuracy is a poor metric for the classifier.** 75% of games clear the 70% bar, so
predicting "yes" for everything scores 75% accuracy and an F1 of 0.858 — higher than some
trained models. ROC-AUC ({M['cls']['roc_auc']:.3f} versus 0.500 for a coin flip) is the
metric that shows real signal.

**Validated forward in time as well.** Training on games released up to 2022 and testing on
2023+ gives R² 0.147, down from {M['reg']['R2']:.3f} on a random split — there is real
distribution shift, with newer games scoring higher on average.
        """
    )
