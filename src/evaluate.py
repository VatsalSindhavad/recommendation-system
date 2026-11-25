from pathlib import Path

import torch
import pandas as pd

from model import NCF

BASE_DIR = Path(__file__).resolve().parents[1]

# ==========================================
# PATHS
# ==========================================

DATA_PATH = BASE_DIR / "data" / "ml-100k" / "u.data"

MODEL_PATH = BASE_DIR / "results" / "ncf_model.pth"

METRICS_PATH = BASE_DIR / "results" / "metrics.txt"

# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

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

# Encode IDs exactly like training

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

print(f"Users: {num_users}")
print(f"Items: {num_items}")

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = NCF(
    num_users=num_users,
    num_items=num_items
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

if isinstance(checkpoint, dict):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully!")

# ==========================================
# HIT@10
# ==========================================

def hit_at_10(model, ratings_df, num_items):

    hits = 0
    evaluated_users = 0

    users = ratings_df["user_id"].unique()

    for user_id in users:

        user_history = ratings_df[
            ratings_df["user_id"] == user_id
        ]

        # Need at least 2 interactions
        if len(user_history) < 2:
            continue

        # Hide one item
        hidden_item = int(
            user_history.iloc[-1]["item_id"]
        )

        visible_history = user_history.iloc[:-1]

        seen_items = set(
            visible_history["item_id"].values
        )

        all_items = torch.arange(
            num_items,
            dtype=torch.long
        )

        user_tensor = torch.full(
            (num_items,),
            int(user_id),
            dtype=torch.long
        )

        with torch.no_grad():

            scores = model(
                user_tensor,
                all_items
            )

        recommendations = [

            (item_id, score.item())

            for item_id, score in enumerate(scores)

            if item_id not in seen_items

        ]

        recommendations.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top10 = [

            item_id

            for item_id, _
            in recommendations[:10]

        ]

        if hidden_item in top10:
            hits += 1

        evaluated_users += 1

    return hits / evaluated_users


# ==========================================
# RUN EVALUATION
# ==========================================

print("\nCalculating Hit@10...")

hit10 = hit_at_10(
    model,
    ratings_df,
    num_items
)

print(f"\nHit@10: {hit10:.4f}")

# ==========================================
# SAVE METRICS
# ==========================================

with open(
    METRICS_PATH,
    "w"
) as f:

    f.write(
        f"Hit@10: {hit10:.4f}\n"
    )

print("\nMetrics saved successfully!")
print(METRICS_PATH)
