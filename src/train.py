from pathlib import Path

import os
import math
import random
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import NCF

BASE_DIR = Path(__file__).resolve().parents[1]

# ==========================================
# LOAD DATA
# ==========================================

print("Loading data...")

ratings_df = pd.read_csv(
    BASE_DIR / "data" / "ml-100k" / "u.data",
    sep="\t",
    names=[
        "user_id",
        "item_id",
        "rating",
        "timestamp"
    ]
)

print("Dataset loaded")

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

print(f"Users: {num_users}")
print(f"Items: {num_items}")

# ==========================================
# USER INTERACTION SETS (for negative sampling)
# ==========================================

user_positive_items = (
    ratings_df
    .groupby("user_id")["item_id"]
    .apply(set)
    .to_dict()
)

# ==========================================
# NORMALIZE RATINGS
# ==========================================

ratings_df["rating"] = (
    ratings_df["rating"] - 3.0
) / 2.0

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

train_df, test_df = train_test_split(
    ratings_df,
    test_size=0.2,
    random_state=42
)

# ==========================================
# DATASET
# ==========================================

class RatingsDataset(Dataset):

    def __init__(self, df):
        self.users = torch.tensor(
            df["user_id"].values,
            dtype=torch.long
        )

        self.items = torch.tensor(
            df["item_id"].values,
            dtype=torch.long
        )

        self.ratings = torch.tensor(
            df["rating"].values,
            dtype=torch.float
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.items[idx],
            self.ratings[idx]
        )

# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    RatingsDataset(train_df),
    batch_size=512,
    shuffle=True
)

test_loader = DataLoader(
    RatingsDataset(test_df),
    batch_size=512
)

# ==========================================
# MODEL
# ==========================================

model = NCF(
    num_users=num_users,
    num_items=num_items
)

print(model)

# ==========================================
# TRAINING SETUP
# ==========================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

epochs = 30
best_val_loss = float("inf")
patience = 7
patience_counter = 0
best_state = None

# Negative sampling config
NUM_NEGATIVES = 2   # negatives per positive interaction
NEG_WEIGHT = 0.3    # weight of negative loss vs positive loss

# ==========================================
# TRAINING LOOP
# ==========================================

print("\nStarting training...\n")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for users, items, ratings in train_loader:

        optimizer.zero_grad()

        batch_size = len(users)

        # ---- Positive loss (MSE on actual ratings) ----
        pos_predictions = model(users, items)
        pos_loss = criterion(pos_predictions, ratings)

        # ---- Negative sampling loss ----
        # For each user in the batch, sample NUM_NEGATIVES
        # items they haven't interacted with.
        # Target for negatives = -1.0 (lowest normalized rating)
        neg_users = users.repeat(NUM_NEGATIVES)
        neg_items = torch.zeros(
            batch_size * NUM_NEGATIVES,
            dtype=torch.long
        )

        for i in range(batch_size):
            u_id = users[i].item()
            positives = user_positive_items.get(u_id, set())
            for n in range(NUM_NEGATIVES):
                neg = random.randint(0, num_items - 1)
                while neg in positives:
                    neg = random.randint(0, num_items - 1)
                neg_items[i + n * batch_size] = neg

        neg_predictions = model(neg_users, neg_items)
        neg_target = torch.full_like(neg_predictions, -1.0)
        neg_loss = criterion(neg_predictions, neg_target)

        # ---- Combined loss ----
        loss = pos_loss + NEG_WEIGHT * neg_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ---- Validation (MSE on held-out ratings only) ----
    model.eval()

    val_loss = 0

    with torch.no_grad():

        for users, items, ratings in test_loader:

            predictions = model(
                users,
                items
            )

            loss = criterion(
                predictions,
                ratings
            )

            val_loss += loss.item()

    val_loss /= len(test_loader)
    val_rmse = math.sqrt(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Val RMSE: {val_rmse:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {
            k: v.clone()
            for k, v in model.state_dict().items()
        }
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(
                f"\nEarly stopping at epoch {epoch+1} "
                f"(no improvement for {patience} epochs)"
            )
            break

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# ==========================================
# EVALUATION
# ==========================================

model.eval()

test_loss = 0

with torch.no_grad():

    for users, items, ratings in test_loader:

        predictions = model(
            users,
            items
        )

        loss = criterion(
            predictions,
            ratings
        )

        test_loss += loss.item()

test_loss /= len(test_loader)

rmse = math.sqrt(test_loss)

print(f"\nTest RMSE: {rmse:.4f}")

# ==========================================
# SAVE MODEL
# ==========================================

results_dir = BASE_DIR / "results"

os.makedirs(
    results_dir,
    exist_ok=True
)

save_path = os.path.join(
    results_dir,
    "ncf_model.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "rmse": rmse,
        "num_users": num_users,
        "num_items": num_items
    },
    save_path
)


print("\nModel saved successfully!")
print("Saved at:")
print(save_path)

print("\nTraining complete!")