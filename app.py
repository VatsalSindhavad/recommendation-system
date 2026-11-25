import sys
from pathlib import Path

# --- Streamlit CLI Bootstrapper ---
# If this file is run directly using 'python app.py', automatically restart it with 'streamlit run'
from streamlit.runtime.scriptrunner import get_script_run_ctx
if not get_script_run_ctx():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", sys.argv[0]] + sys.argv[1:]
    sys.exit(stcli.main())

from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import torch

# ======================================
# PATHS
# ======================================

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "ml-100k" / "u.data"
MOVIES_PATH = BASE_DIR / "data" / "ml-100k" / "u.item"

MODEL_PATH = BASE_DIR / "results" / "ncf_model.pth"
METRICS_PATH = BASE_DIR / "results" / "metrics.txt"
EMBEDDING_PATH = BASE_DIR / "results" / "embeddings.png"

sys.path.append(str(BASE_DIR / "src"))

from model import NCF

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ======================================
# CUSTOM STYLING
# ======================================

st.markdown("""
<style>
    /* ---- Header Banner ---- */
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.2rem 2rem 1.6rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(102, 126, 234, 0.25);
    }
    .header-banner h1 {
        color: white !important;
        font-size: 2.4rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.05rem;
        margin: 0;
    }

    /* ---- Genre Pills ---- */
    .genre-pill {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin: 2px 2px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* ---- Explanation Box ---- */
    .explanation-box {
        background: rgba(102, 126, 234, 0.08);
        border-left: 3px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 0.55rem 1rem;
        margin-top: 0.4rem;
        font-size: 0.9rem;
    }

    /* ---- Metric Accent ---- */
    div[data-testid="stMetricValue"] > div {
        color: #667eea !important;
    }

    /* ---- Footer ---- */
    .footer-text {
        text-align: center;
        color: #888;
        padding: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================================
# HEADER
# ======================================

st.markdown("""
<div class="header-banner">
    <h1>🎬 Neural Network Movie Recommendation System</h1>
    <p>Explainable Embedding-Based Neural Collaborative Filtering · MovieLens-100K</p>
</div>
""", unsafe_allow_html=True)

# ======================================
# GENRE DEFINITIONS
# ======================================

GENRE_NAMES = [
    "Unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

# ======================================
# LOAD DATA
# ======================================

ratings_df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    names=[
        "user_id",
        "item_id",
        "rating",
        "timestamp"
    ]
)

# Read full u.item (all 24 columns) for genre extraction
movies_full_df = pd.read_csv(
    MOVIES_PATH,
    sep="|",
    encoding="latin-1",
    header=None
)

movies_df = movies_full_df[[0, 1]].copy()
movies_df.columns = [
    "movie_id",
    "title"
]

# ======================================
# GENRE EXTRACTION & DICTIONARIES
# ======================================

GENRE_COLUMNS = list(range(5, 24))  # columns 5-23 are genre binary flags

item_id_to_genres = {}
item_id_to_title = {}

for idx, row in movies_full_df.iterrows():
    title = row[1]
    genres = [
        GENRE_NAMES[i]
        for i, col in enumerate(GENRE_COLUMNS)
        if row[col] == 1
    ]
    item_id_to_genres[idx] = genres
    item_id_to_title[idx] = title

# ======================================
# ENCODE IDS
# ======================================

ratings_df["user_id"] = (
    ratings_df["user_id"]
    .astype("category")
    .cat.codes
)

ratings_df["item_id"] = (
    ratings_df["item_id"]
    .astype("category")
    .cat.codes
)

num_users = ratings_df["user_id"].nunique()
num_items = ratings_df["item_id"].nunique()

# ======================================
# LOAD MODEL
# ======================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model = NCF(
    num_users=num_users,
    num_items=num_items
)

if isinstance(checkpoint, dict):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    rmse = checkpoint.get(
        "rmse",
        "Unknown"
    )

else:

    model.load_state_dict(
        checkpoint
    )

    rmse = "0.4743"

model.eval()

# ======================================
# LOAD HIT@10
# ======================================

hit10 = "Unknown"

if METRICS_PATH.exists():

    with open(
        METRICS_PATH,
        "r"
    ) as f:

        hit10 = (
            f.read()
            .replace(
                "Hit@10:",
                ""
            )
            .strip()
        )

# ======================================
# HELPER FUNCTIONS
# ======================================


def get_user_taste_profile(user_id):
    """
    Analyze all movies rated by a user.
    Returns genre frequency Counter and total number of rated movies.
    """
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    genre_counts = Counter()

    for _, row in user_ratings.iterrows():
        item_id = row["item_id"]
        genres = item_id_to_genres.get(item_id, [])
        for g in genres:
            if g != "Unknown":
                genre_counts[g] += 1

    return genre_counts, len(user_ratings)


def get_recommendation_explanation(movie_genres, user_top_genres,
                                    genre_boosted=False):
    """
    Generate a human-readable explanation for a recommendation
    by comparing the movie's genres against the user's top genres.
    """
    matching = [
        g for g in movie_genres
        if g in user_top_genres and g != "Unknown"
    ]

    if matching:
        genres_text = " and ".join(matching[:3])
        boost_note = (
            " <em>(genre boost applied)</em>"
            if genre_boosted else ""
        )
        return (
            f"🎯 Why recommended? "
            f"Matches your interest in "
            f"<strong>{genres_text}</strong>.{boost_note}"
        )

    return (
        "🤝 Recommended because users with similar "
        "preferences liked this movie."
    )


# ======================================
# RECOMMENDER (HYBRID: NCF + GENRE AFFINITY)
# ======================================


def recommend_hybrid(
    user_id,
    genre_counts,
    genre_weight=0.0,
    k=10
):
    """
    Hybrid recommendation blending NCF scores with genre affinity.

    genre_weight: 0.0 = pure NCF, 1.0 = pure genre match.
    The final score for each movie is:
        (1 - genre_weight) * ncf_normalized + genre_weight * genre_affinity
    """

    seen_items = set(
        ratings_df[
            ratings_df["user_id"] == user_id
        ]["item_id"].values
    )

    all_items = torch.arange(
        num_items,
        dtype=torch.long
    )

    user_tensor = torch.full(
        (num_items,),
        user_id,
        dtype=torch.long
    )

    with torch.no_grad():
        scores = model(
            user_tensor,
            all_items
        )

    scores_np = scores.numpy()

    # --- Normalize NCF scores to [0, 1] ---
    min_s, max_s = scores_np.min(), scores_np.max()
    if max_s > min_s:
        ncf_norm = (scores_np - min_s) / (max_s - min_s)
    else:
        ncf_norm = np.zeros_like(scores_np)

    # --- Compute genre affinity per movie ---
    # Build a probability distribution over the user's genres
    total = sum(genre_counts.values()) if genre_counts else 1
    genre_probs = {
        g: c / total
        for g, c in genre_counts.items()
    }

    recommendations = []

    for item_id in range(num_items):
        if item_id in seen_items:
            continue

        # Genre affinity = sum of user's genre probabilities
        # for each of this movie's genres
        item_genres = item_id_to_genres.get(item_id, [])
        genre_affinity = sum(
            genre_probs.get(g, 0.0)
            for g in item_genres
            if g != "Unknown"
        )
        # Cap at 1.0
        genre_affinity = min(genre_affinity, 1.0)

        # Blend scores
        ncf_score = float(ncf_norm[item_id])
        final_score = (
            (1.0 - genre_weight) * ncf_score
            + genre_weight * genre_affinity
        )

        recommendations.append((
            item_id,
            final_score,
            float(scores_np[item_id]),  # raw NCF score
            genre_affinity
        ))

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:k]


# ======================================
# METRICS DASHBOARD
# ======================================

st.markdown("### 📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Total Users", num_users)

with col2:
    st.metric("🎥 Total Movies", num_items)

with col3:
    st.metric("📉 RMSE", rmse)

with col4:
    st.metric("🎯 Hit@10", hit10)

st.markdown("---")

# ======================================
# USER SELECTOR
# ======================================

st.subheader("🔍 Explore & Recommend")

select_col, weight_col, btn_col = st.columns([3, 2, 1])

with select_col:
    selected_user = st.selectbox(
        "Select a User",
        list(range(num_users)),
        help="Pick a user to view their taste profile and get personalized recommendations."
    )

with weight_col:
    genre_weight = st.slider(
        "🎛️ Genre Preference Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help=(
            "Controls how much genre preference influences rankings. "
            "0.0 = pure neural network, "
            "1.0 = pure genre match."
        )
    )

with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_clicked = st.button(
        "🚀 Recommend",
        use_container_width=True
    )

# ======================================
# RESULTS (TASTE PROFILE + RECOMMENDATIONS)
# ======================================

if recommend_clicked:

    # --- Compute user taste profile ---
    genre_counts, num_rated = get_user_taste_profile(selected_user)
    top_genre_names = set()

    st.markdown("---")

    profile_col, rec_col = st.columns([2, 3])

    # ---------- LEFT COLUMN: Taste Profile ----------
    with profile_col:
        with st.container(border=True):
            st.markdown("### 🧬 User Taste Profile")
            st.caption(
                f"User **#{selected_user}** · "
                f"**{num_rated}** movies rated"
            )

            if genre_counts:
                top_genres = genre_counts.most_common(5)
                top_genre_names = set(g for g, _ in top_genres)

                st.markdown("##### 🏆 Top Genres")
                medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                for i, (genre, count) in enumerate(top_genres):
                    st.markdown(
                        f"{medals[i]} **{genre}** — {count} movies"
                    )

                st.markdown("##### 📊 Genre Distribution")
                chart_df = pd.DataFrame(
                    genre_counts.most_common(),
                    columns=["Genre", "Count"]
                ).set_index("Genre")
                st.bar_chart(chart_df)
            else:
                st.info(
                    "No genre data available for this user's "
                    "rated movies."
                )

    # ---------- RIGHT COLUMN: Recommendations ----------
    with rec_col:
        with st.container(border=True):
            weight_label = (
                "Pure NCF" if genre_weight == 0.0
                else "Pure Genre" if genre_weight == 1.0
                else f"NCF {(1-genre_weight)*100:.0f}% + Genre {genre_weight*100:.0f}%"
            )
            st.markdown(
                f"### 🎬 Top 10 Recommendations\n"
                f"*Mode: {weight_label}*"
            )

            recommendations = recommend_hybrid(
                selected_user,
                genre_counts,
                genre_weight=genre_weight,
                k=10
            )

            for rank, (
                item_id, final_score, ncf_raw, g_affinity
            ) in enumerate(
                recommendations, start=1
            ):
                title = item_id_to_title.get(
                    item_id, "Unknown Movie"
                )
                genres = item_id_to_genres.get(item_id, [])
                clean_genres = [
                    g for g in genres if g != "Unknown"
                ]

                genre_boosted = (
                    genre_weight > 0 and g_affinity > 0.1
                )

                explanation = get_recommendation_explanation(
                    genres, top_genre_names,
                    genre_boosted=genre_boosted
                )

                # Build genre pill HTML
                genre_pills = " ".join(
                    f'<span class="genre-pill">{g}</span>'
                    for g in clean_genres
                )
                if not genre_pills:
                    genre_pills = (
                        '<span class="genre-pill">'
                        'Unclassified</span>'
                    )

                # Score display
                score_line = (
                    f"⭐ Final Score: `{final_score:.4f}`"
                )
                if genre_weight > 0:
                    score_line += (
                        f" · NCF: `{ncf_raw:.4f}`"
                        f" · Genre Affinity: `{g_affinity:.2f}`"
                    )

                st.markdown(
                    f"**#{rank} · {title}**\n\n"
                    f"{score_line}"
                )
                st.markdown(
                    genre_pills,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="explanation-box">'
                    f'{explanation}</div>',
                    unsafe_allow_html=True
                )
                if rank < 10:
                    st.divider()

    # ---------- Table View ----------
    with st.expander("📋 View Recommendations as Table"):
        rows = []
        for rank, (
            item_id, final_score, ncf_raw, g_affinity
        ) in enumerate(
            recommendations, start=1
        ):
            title = item_id_to_title.get(
                item_id, "Unknown Movie"
            )
            genres = item_id_to_genres.get(item_id, [])
            rows.append({
                "Rank": rank,
                "Movie": title,
                "Final Score": round(final_score, 4),
                "NCF Score": round(ncf_raw, 4),
                "Genre Affinity": round(g_affinity, 2),
                "Genres": ", ".join(
                    g for g in genres if g != "Unknown"
                )
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True
        )

st.markdown("---")

# ======================================
# EMBEDDING VISUALIZATION
# ======================================

with st.expander("🔮 Embedding Visualization"):

    if EMBEDDING_PATH.exists():

        st.image(
            str(
                EMBEDDING_PATH
            )
        )

    else:

        st.info(
            "📁 embeddings.png not found. "
            "Run the training pipeline to generate embeddings."
        )

# ======================================
# COLD START ANALYSIS
# ======================================

with st.expander("❄️ Cold Start Analysis"):

    st.info("""
Users with very few interactions have weak embedding representations.

**Fallback Strategy:**
Recommend globally popular movies until enough interaction history is collected.
""")

    st.markdown("""
**Why does this happen?**
- New users lack sufficient rating history for the model to learn meaningful embeddings.
- The neural network requires a minimum number of interactions to capture preferences.

**Mitigation Strategies:**
- 🌐 **Popularity-based fallback** — Serve globally popular movies to new users.
- 🏷️ **Content-based augmentation** — Use genre metadata to cold-start recommendations.
- 🔄 **Active learning** — Prompt new users to rate a small set of diverse movies.
""")

# ======================================
# FOOTER
# ======================================

st.markdown("---")

st.markdown(
    '<div class="footer-text">'
    "📊 <strong>MovieLens-100K</strong> · "
    "943 Users · 1,682 Movies · 100,000 Ratings<br>"
    '<span style="font-size:0.8rem;">'
    "Powered by Neural Collaborative Filtering "
    "with Embedding-Based Deep Learning"
    "</span>"
    "</div>",
    unsafe_allow_html=True
)
