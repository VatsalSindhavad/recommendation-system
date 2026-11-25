from pathlib import Path

import torch
import pandas as pd

from model import NCF

BASE_DIR = Path(__file__).resolve().parents[1]

# ==========================================
# PATHS
# ==========================================

DATA_PATH = BASE_DIR / "data" / "ml-100k" / "u.data"

MOVIES_PATH = BASE_DIR / "data" / "ml-100k" / "u.item"

MODEL_PATH = BASE_DIR / "results" / "ncf_model.pth"

# ==========================================
# LOAD DATA
# ==========================================

print("Loading data...")

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

movies_df = pd.read_csv(
    MOVIES_PATH,
    sep="|",
    encoding="latin-1",
    header=None
)

movies_df = movies_df[[0, 1]]
movies_df.columns = [
    "movie_id",
    "title"
]

# ==========================================
# ENCODE USERS / ITEMS
# ==========================================

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

# ==========================================
# ITEM -> TITLE MAPPING
# ==========================================

item_id_to_title = {
    idx: title
    for idx, title in enumerate(
        movies_df["title"]
    )
}

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = NCF(
    num_users=num_users,
    num_items=num_items
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully!")

# ==========================================
# RECOMMENDER
# ==========================================

def recommend_top_k(
    model,
    ratings_df,
    user_id,
    k=10
):

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

    item_scores = [

        (item_id, score.item())

        for item_id, score in enumerate(scores)

        if item_id not in seen_items

    ]

    item_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return item_scores[:k]

# ==========================================
# EXAMPLE RECOMMENDATIONS
# ==========================================

user_id = 0

recommendations = recommend_top_k(
    model,
    ratings_df,
    user_id,
    k=10
)

print(f"\nTop 10 Recommendations for User {user_id}\n")

for item_id, score in recommendations:

    title = item_id_to_title.get(
        item_id,
        "Unknown Movie"
    )

    print(
        f"{title} | Predicted Score: {score:.4f}"
    )