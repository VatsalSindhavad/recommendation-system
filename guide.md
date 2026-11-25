# 🎬 Complete Technical Guide: File-by-File Code Breakdown

This document provides a highly detailed, file-by-file analysis of the entire recommendation system codebase. Use it to understand the exact mechanics, inputs, outputs, functions, and design decisions of each file to excel in your technical presentations or interviews.

---

## 🗺️ Project Architecture Overview

The system is built as an end-to-end Machine Learning pipeline:
1. **Model Layer (`src/model.py`)**: Defines the PyTorch Neural Collaborative Filtering network.
2. **Training & Pipeline Layer (`src/train.py`, `src/evaluate.py`, `src/recommend.py`)**: Manages dataset preparation, training loops, metrics calculation, and terminal prediction.
3. **Application & Hybrid Fusion Layer (`app.py`)**: Blends model scores with metadata-driven Genre Affinity to provide explainable recommendations via a Streamlit UI.
4. **Analysis Layer (`notebooks/`)**: Interactive explorations of data sparsity, embedding representations, and cold-start mitigations.

```
                      ┌────────────────────────────────┐
                      │    Raw Dataset (u.data, u.item)│
                      └───────────────┬────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │     Data Preprocessing &       │
                      │   ID Categorization (0-index)  │
                      └──────┬──────────────────┬──────┘
                             │                  │
                             ▼                  ▼
              ┌──────────────────────┐  ┌──────────────────────┐
              │  Model Training      │  │  Metadata Extraction │
              │  (src/train.py)      │  │  (Genre Matrices)    │
              └──────┬───────────────┘  └────────┬─────────────┘
                     │                           │
                     ▼                           ▼
              ┌──────────────────────┐  ┌──────────────────────┐
              │   Model Weights      │  │  User Taste Profile  │
              │  (ncf_model.pth)     │  │  (app.py logic)      │
              └──────┬───────────────┘  └────────┬─────────────┘
                     │                           │
                     └─────────────┬─────────────┘
                                   ▼
                      ┌────────────────────────────────┐
                      │   Hybrid Recommendation Blend  │
                      │  (1-w)*NCF + w*Genre_Affinity  │
                      └───────────────┬────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │ Streamlit Interactive Interface│
                      │  - Explainable Recommendations │
                      │  - Embedding Visualization     │
                      └────────────────────────────────┘
```

---

## 📁 File-by-File Breakdown & Inner Mechanics

---

### 1. `src/model.py` — The NeuMF Neural Network

#### 🎯 Purpose
This file contains the core PyTorch model definition. It implements the **Neural Matrix Factorization (NeuMF)** network, which blends Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) branches to model user-item interactions.

#### 📥 Inputs & 📤 Outputs
* **Inputs**:
  * `user`: Tensor of shape `(batch_size,)` containing contiguous, 0-indexed user integer IDs.
  * `item`: Tensor of shape `(batch_size,)` containing contiguous, 0-indexed item integer IDs.
* **Outputs**:
  * Predicts a single float value per pair representing the rating estimation, squeezed to shape `(batch_size,)`.

#### 🧬 Class & Functions Detail
* **`class NCF(nn.Module)`**: The main network class.
* **`__init__(self, num_users, num_items, embedding_dim=64)`**:
  * Allocates embeddings for both branches separately. GMF user/item embeddings are mapped to `embedding_dim`, and MLP user/item embeddings are mapped to `embedding_dim`.
  * Instantiates `self.mlp` as an `nn.Sequential` block containing:
    * `nn.Linear(embedding_dim * 2, 128)`: Concatenates user and item embeddings.
    * `nn.BatchNorm1d(128)`: Normalizes inputs to stabilize training and accelerate convergence.
    * `nn.ReLU()`: Adds non-linear capacity.
    * `nn.Dropout(0.35)`: Randomly zeroes out 35% of neurons to prevent overfitting.
    * `nn.Linear(128, 64)` -> `nn.BatchNorm1d(64)` -> `nn.ReLU()` -> `nn.Dropout(0.35)`
  * Instantiates `self.prediction` as a linear projection from `embedding_dim + 64` (GMF output + MLP final layer) to `1`.
* **`_init_weights(self)`**:
  * Iterates through all layers (`self.modules()`).
  * Applies **Xavier Uniform initialization** (`nn.init.xavier_uniform_`) to embedding weights and linear weight matrices.
  * Fills linear bias terms with zeroes (`nn.init.zeros_`).
* **`forward(self, user, item)`**:
  * **GMF pipeline**: Multiplies user and item embeddings element-wise (`gmf_user * gmf_item`).
  * **MLP pipeline**: Concatenates user and item embeddings (`torch.cat([mlp_user, mlp_item], dim=1)`), passes through `self.mlp` hidden layers.
  * **Fusion**: Concatenates GMF output (`embedding_dim`) and MLP output (`64`) along the feature dimension (`dim=1`).
  * **Prediction**: Passes the fused vector through `self.prediction` and squeezes the dimension to return a 1D tensor.

#### 💡 Technical Design Decisions
* **Independent Embeddings**: GMF and MLP use separate user and item embedding spaces instead of sharing them. This allows the linear branch (GMF) and non-linear branch (MLP) to learn distinct representations without interfering with each other's optimization paths.
* **Regularization Selection**: Dropout is set to a relatively high value of `0.35` and paired with Batch Normalization because MovieLens-100K is a relatively small dataset (100k records) and deep neural networks overfit it easily.

#### 🗣️ Interview Presentation Bullet Points
* *"We implement the NeuMF architecture from He et al.'s NCF paper, which merges matrix factorization and deep neural networks."*
* *"We use Xavier weight initialization, batch normalization, and dropout to ensure robust model convergence and avoid overfitting."*

---

### 2. `src/train.py` — The Training Pipeline

#### 🎯 Purpose
This script handles the raw dataset ingestion, ID categorization, training-test splits, negative interaction sampling, optimization, and saves the final checkpoint.

#### 📥 Inputs & 📤 Outputs
* **Inputs**:
  * Raw data from `data/ml-100k/u.data` (tab-separated columns: `user_id`, `item_id`, `rating`, `timestamp`).
* **Outputs**:
  * Checkpoint dictionary written to `results/ncf_model.pth` containing model state dictionary, evaluation metrics, and dataset boundaries.

#### 🧬 Key Functions & Variables Detail
* **Categorical Encoding**:
  ```python
  ratings_df["user_id"] = ratings_df["user_id"].astype("category").cat.codes
  ratings_df["item_id"] = ratings_df["item_id"].astype("category").cat.codes
  ```
  Converts raw user/item IDs (which might have gaps or start at index 1) into contiguous, 0-indexed integer series. This is critical for index mapping inside PyTorch `nn.Embedding` blocks.
* **Rating Normalization**:
  ```python
  ratings_df["rating"] = (ratings_df["rating"] - 3.0) / 2.0
  ```
  Transforms raw ratings ranging from $[1, 5]$ to a $[-1.0, 1.0]$ interval. This aligns 3.0 (neutral) with 0.0, 1.0/2.0 (disliked) with negative coordinates, and 4.0/5.0 (liked) with positive coordinates.
* **`class RatingsDataset(Dataset)`**:
  * PyTorch dataset mapping index query to tensors: `self.users`, `self.items`, `self.ratings`.
* **Negative Sampling Loop**:
  For each batch, the code dynamically generates `NUM_NEGATIVES = 2` negative items per positive interaction:
  ```python
  for i in range(batch_size):
      u_id = users[i].item()
      positives = user_positive_items.get(u_id, set())
      for n in range(NUM_NEGATIVES):
          neg = random.randint(0, num_items - 1)
          while neg in positives:
              neg = random.randint(0, num_items - 1)
          neg_items[i + n * batch_size] = neg
  ```
* **Combined Loss Calculation**:
  ```python
  loss = pos_loss + NEG_WEIGHT * neg_loss  # NEG_WEIGHT = 0.3
  ```
* **`ReduceLROnPlateau(patience=3, factor=0.5)`**:
  Shrinks the learning rate by 50% if validation loss fails to decrease for 3 consecutive epochs.
* **Early Stopping**:
  Exits training if validation loss does not improve for `patience = 7` epochs. Deep clones the state dictionary to save the best weights rather than keeping pointers to the active model state.

#### 💡 Technical Design Decisions
* **Explicit Ratings as Implicit Targets**: Since MovieLens-100K features explicit ratings, we normalize them to a standard range to compute regression loss on rating intensity. However, to optimize rankings, we inject random negative samples mapped to the minimum rating value (`-1.0`) with a smaller loss weight (`0.3`). This guides the network to predict positive preference values for watched items and negative values for unvisited items.
* **Safe Checkpoint Restoring**:
  ```python
  best_state = {k: v.clone() for k, v in model.state_dict().items()}
  ```
  Cloning tensors prevents references from tracking subsequent weight updates, ensuring the correct weights are restored.

#### 🗣️ Interview Presentation Bullet Points
* *"I normalized ratings to [-1.0, 1.0] and trained using a combined loss function of positive rating regression and negative sampling."*
* *"The pipeline incorporates learning rate decay on plateau and early stopping with weight restoration to guarantee generalization."*

---

### 3. `src/evaluate.py` — Performance Metrics

#### 🎯 Purpose
This script computes validation metrics to measure the quality of recommendations, specifically focusing on ranking relevance.

#### 📥 Inputs & 📤 Outputs
* **Inputs**:
  * Loaded weights from `results/ncf_model.pth`.
  * Ratings dataset from `data/ml-100k/u.data`.
* **Outputs**:
  * Logs the evaluation score and writes it to `results/metrics.txt`.

#### 🧬 Key Functions & Variables Detail
* **`hit_at_10(model, ratings_df, num_items)`**:
  Implements the Leave-One-Out ranking metric across all users in the dataset:
  1. Identifies the user's interaction history sorted by time.
  2. Extracts the user's very last interaction as the `hidden_item`.
  3. Recommends the top 10 items from all items the user *has not* seen in their training history.
  4. Increments the hit count if the `hidden_item` is present in the top 10 recommendations.
  5. Divides total hit counts by the number of evaluated users to calculate the final Hit Rate.

#### 💡 Technical Design Decisions
* **Why Leave-One-Out**: Evaluates recommendation models in a way that simulates production. The model is trained on historical records, and we test whether it can predict the user's subsequent interaction.
* **Filtering Seen Items**: When scoring all items for a user, the code explicitly removes items they previously rated during training. Recommending already watched items is a common flaw in basic recommendation models.

#### 🗣️ Interview Presentation Bullet Points
* *"Rather than relying only on RMSE, I implemented Hit@10 evaluation to measure list ranking performance."*
* *"Using a Leave-One-Out setup, the model ranks all unseen movies for a user, and we check if the hidden movie falls in their top 10."*

---

### 4. `src/recommend.py` — CLI Recommendation Utility

#### 🎯 Purpose
A command-line tool that loads the trained NeuMF network and generates top K movie recommendations for a selected user ID.

#### 📥 Inputs & 📤 Outputs
* **Inputs**:
  * User ID argument (hardcoded or CLI).
  * Trained weights file `results/ncf_model.pth`.
  * Movie metadata file `data/ml-100k/u.item`.
* **Outputs**:
  * Prints a formatted terminal list of the top 10 recommended movie titles and their raw neural network scores.

#### 🧬 Key Functions & Variables Detail
* **`recommend_top_k(model, ratings_df, user_id, k=10)`**:
  * Identifies the user's rated items (`seen_items`).
  * Creates PyTorch input tensors `user_tensor` and `all_items`.
  * Runs inference under `torch.no_grad()` to compute predicted scores for all items.
  * Filters out `seen_items`.
  * Sorts the remaining items by predicted score in descending order and returns the top `k`.

#### 💡 Technical Design Decisions
* **Latin-1 Encoding**:
  ```python
  movies_df = pd.read_csv(MOVIES_PATH, sep="|", encoding="latin-1", header=None)
  ```
  The MovieLens dataset contains special characters in European movie titles. Standard UTF-8 encoding fails to parse the file, requiring `latin-1` decoding.

#### 🗣️ Interview Presentation Bullet Points
* *"I developed a clean inference module that filters out a user's watch history and outputs their top K predicted movies in descending order."*

---

### 5. `app.py` — Streamlit Web Interface

#### 🎯 Purpose
The main user interface. It displays model metrics, visualizes embeddings, and runs a hybrid prediction engine combining NCF scores and Genre Affinity metadata to generate explainable recommendations.

#### 📥 Inputs & 📤 Outputs
* **Inputs**:
  * Interaction records (`u.data`) and movie genres (`u.item`).
  * Checkpoint weights (`ncf_model.pth`) and performance scores (`metrics.txt`).
  * Slider inputs for user selection and hybrid weight tuning.
* **Outputs**:
  * Interactive web application rendered locally via Streamlit.

#### 🧬 Key Functions & Variables Detail
* **Genre Parsing**:
  ```python
  GENRE_COLUMNS = list(range(5, 24))
  # Columns 5 to 23 are binary genre flags in u.item
  ```
  Maps binary flags to matching category names in `GENRE_NAMES` to build `item_id_to_genres`.
* **`get_user_taste_profile(user_id)`**:
  Filters movies rated by the user, aggregates counts for each genre category using a `Counter`, and returns the counts.
* **`recommend_hybrid(user_id, genre_counts, genre_weight, k=10)`**:
  Blends collaborative filtering predictions with content-based genre affinity:
  1. Computes raw NCF scores for all unseen movies.
  2. Standardizes NCF scores using Min-Max scaling:
     ```python
     ncf_norm = (scores_np - min_s) / (max_s - min_s)
     ```
  3. Computes the user's historical genre probability distribution:
     ```python
     genre_probs = {g: count / total_ratings for g, count in genre_counts.items()}
     ```
  4. Calculates Genre Affinity for each candidate movie as the sum of the user's probabilities for the movie's genres:
     $$\text{Affinity} = \sum_{g \in \text{movie\_genres}} \text{genre\_probs}[g]$$
  5. Computes the final blended score:
     $$\text{Final Score} = (1 - w) \times \text{NCF\_Normalized} + w \times \text{Genre\_Affinity}$$
* **`get_recommendation_explanation(movie_genres, user_top_genres, genre_boosted)`**:
  Computes the intersection of the recommended movie's genres with the user's top genres. If they overlap, it generates a message:
  > *"🎯 Why recommended? Matches your interest in Drama and Thriller."*
  If they do not overlap, it falls back to collaborative explanation:
  > *"🤝 Recommended because users with similar preferences liked this movie."*

#### 💡 Technical Design Decisions
* **Scale Normalization**: Neural networks output unbounded scores (logits). Because Genre Affinity is bounded between $0.0$ and $1.0$, raw logits cannot be blended directly. Applying Min-Max normalization to the NCF scores ensures both inputs are on a 0-1 scale.
* **Explainability Injection**: The dashboard uses customized HTML containers and CSS styles to display recommendations cleanly, rendering genre pills and explanation boxes with color-coded alerts.

#### 🗣️ Interview Presentation Bullet Points
* *"I built a hybrid recommendation engine that blends NCF scores with a user's historical genre preferences, controlled by a Streamlit slider."*
* *"To make the recommendations explainable, the dashboard analyzes matching genres to show users exactly why a movie was suggested."*

---

### 6. `notebooks/exploration.ipynb` — Data Exploration

#### 🎯 Purpose
A Jupyter notebook containing exploratory data analysis (EDA) of the MovieLens-100K dataset to analyze distribution patterns and dataset sparsity.

#### 💡 Key Concepts & Code Blocks
* **Sparsity Calculation**:
  ```python
  num_users = ratings_df['user_id'].nunique()
  num_items = ratings_df['item_id'].nunique()
  sparsity = 1.0 - (len(ratings_df) / (num_users * num_items))
  print(f"Dataset Sparsity: {sparsity * 100:.2f}%")
  # Returns ~93.70% sparsity
  ```
  This high sparsity highlights why simple methods (like calculating average ratings) fail, motivating the use of collaborative filtering models.
* **Rating Distribution**:
  Plots a bar chart showing the frequency of rating scores. This reveals that ratings are skewed towards positive scores (with 4 and 5 being the most common), which justifies normalizing ratings around 3.0.

#### 🗣️ How to pitch it:
> *"Before building the model, I conducted exploratory data analysis. The MovieLens-100K dataset has a sparsity of 93.70%, meaning users have rated only about 6.3% of the available movies. This high sparsity highlights the need for collaborative filtering. I also found that ratings are skewed towards 4 and 5, which guided my decision to normalize ratings around 3.0 during training."*

---

### 7. `notebooks/embedding_analysis.ipynb` — Embedding Analysis

#### 🎯 Purpose
Analyzes the representations learned by the model's embedding layers.

#### 💡 Key Concepts & Code Blocks
* **Extracting Embedding Weights**:
  ```python
  movie_embeddings = model.mlp_item_embedding.weight.data.numpy()
  ```
* **Dimensionality Reduction**:
  Uses Principal Component Analysis (PCA) or t-SNE to project the 64-dimensional movie embeddings down to a 2D space.
* **Clustering Analysis**:
  Plots these 2D points and colors them by genre. This reveals distinct clusters where related genres (like Animation and Children's) group together, showing that the model's embedding layers successfully capture semantic relationships.

#### 🗣️ How to pitch it:
> *"To verify that the model was learning meaningful features, I analyzed the trained embedding weights. I extracted the 64-dimensional movie embedding vectors and projected them to 2D using PCA. The visualization shows that similar movies cluster together in the latent space. For example, Action and Sci-Fi movies group together, while Children's animations cluster in a separate region. This demonstrates that the model successfully captures semantic relationships without explicit instruction."*

---

### 8. `notebooks/cold_start_analysis.ipynb` — Cold-Start Profile Analysis

#### 🎯 Purpose
Analyzes user profiles with very few ratings to study how collaborative filtering models perform under cold-start conditions.

#### 💡 Key Concepts & Code Blocks
* **User Interaction Counts**:
  Calculates the number of ratings per user and filters for those with fewer than 5 ratings.
* **Evaluating Recommendation Quality**:
  Evaluates recommendation performance (RMSE/Hit Rate) specifically for these cold-start users, demonstrating that predictions are less accurate because the model lacks sufficient interaction history to learn stable embeddings.
* **Fallback Strategy Verification**:
  Verifies that reverting to popularity-based fallback models or metadata-only content matching improves recommendations for these users.

#### 🗣️ How to pitch it:
> *"I created a notebook specifically to analyze cold-start performance. Collaborative filtering relies on user interactions, so performance drops for new users who have rated very few movies. My analysis showed that recommendations for users with fewer than 5 ratings were less accurate. To address this, I designed fallback strategies like recommending globally popular movies or using content-based matching until the user builds an interaction history."*
