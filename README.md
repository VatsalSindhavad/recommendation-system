# 🎬 Neural Collaborative Filtering (NCF) Movie Recommendation System

An end-to-end, explainable movie recommendation system trained on the **MovieLens-100K** dataset. The system features a custom **Neural Matrix Factorization (NeuMF)** deep learning model built with PyTorch, a hybrid blending engine, and an interactive Streamlit dashboard.

---

## 🚀 Key Features

* **NeuMF Deep Learning Model**: Implements Neural Matrix Factorization fusing Generalized Matrix Factorization (GMF) for linear interactions and Multi-Layer Perceptron (MLP) for non-linear interactions.
* **Hybrid Recommendation Engine**: Blends neural collaborative filtering signals with content-based **Genre Affinity** profiles, controllable at runtime via an interactive slider.
* **Explainable Recommendations**: Generates natural language rationales for recommended movies by matching movie genres against the user's top historical genres.
* **User Taste Profiling**: Analyzes user activity to compute and display top genres and a distribution chart of their historical ratings.
* **Comprehensive Metrics**: Evaluates performance using both rating prediction error (**RMSE**) and list ranking quality (**Hit@10** via leave-one-out evaluation).
* **Embedding Visualization**: Visualizes high-dimensional user and item embeddings in a low-dimensional space to show learned preferences.

---

## 📐 Model Architecture

The core recommender uses **Neural Matrix Factorization (NeuMF)**, combining two complementary embedding pipelines:
1. **GMF (Generalized Matrix Factorization)**: Multiplies user and item embedding dimensions element-wise to capture linear relationships.
2. **MLP (Multi-Layer Perceptron)**: Concatenates user and item embeddings and processes them through fully connected layers with batch normalization, ReLU activation, and dropout to capture non-linear relationships.

The outputs are concatenated and projected to predict the preference score:

```
                  User ID           Item ID
                     │                 │
           ┌─────────┴─────────┐       ├───────────────────┐
           ▼                   ▼       ▼                   ▼
     GMF Embedding       MLP Embedding   GMF Embedding       MLP Embedding
           │                   │       │                   │
           └─────────┬─────────┘       └─────────┬─────────┘
                     ▼                           ▼
          GMF (Element-wise Prod)     MLP (Concatenation)
                     │                           │
                     │                           ▼
                     │                 MLP Dense Layer (128)
                     │                           ▼
                     │                 MLP Dense Layer (64)
                     │                           │
                     └─────────────┬─────────────┘
                                   ▼
                             Concatenation
                                   ▼
                         Final Prediction Layer
                                   ▼
                            Predicted Score
```

---

## 📁 Project Structure

```
recommendation-system/
├── data/                         # Dataset directory
│   └── ml-100k/                  # MovieLens-100k data (u.data, u.item, etc.)
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── cold_start_analysis.ipynb # Visualizing cold start user profiles
│   ├── embedding_analysis.ipynb  # Dimension reduction of learned embeddings
│   └── exploration.ipynb         # Exploratory data analysis
├── results/                      # Trained model weights & outputs
│   ├── embeddings.png            # Visual cluster plot of item embeddings
│   ├── metrics.txt               # Evaluation output (Hit@10)
│   └── ncf_model.pth             # PyTorch saved weights
├── src/                          # Model source code
│   ├── evaluate.py               # Evaluates model Hit@10 performance
│   ├── model.py                  # PyTorch NeuMF model class
│   ├── recommend.py              # CLI recommendations generator
│   └── train.py                  # Training pipeline with negative sampling
├── app.py                        # Streamlit dashboard & Hybrid app
├── requirements.txt              # Required packages
└── README.md                     # Project documentation (this file)
```

---

## 🛠️ Installation & Setup

### 1. Clone & Navigate to Workspace
```bash
git clone https://github.com/VatsalSindhavad/recommendation-system.git
cd recommendation-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Requirements: `torch`, `pandas`, `numpy`, `streamlit`, `scikit-learn`, `matplotlib`)*

### 3. Extract the Dataset
Download MovieLens 100K from the GroupLens link below and extract the files into the `data/ml-100k/` directory. The folders should contain `u.data` (interactions) and `u.item` (movie metadata).

---

## 🔄 How to Run

### 1. Train the Neural Network
Trains the NeuMF network on the user interaction data using negative sampling, early stopping, and learning rate scheduling:
```bash
python src/train.py
```
This saves the trained weights to `results/ncf_model.pth`.

### 2. Run Model Evaluation
Runs Leave-One-Out Hit@10 validation to evaluate ranking performance:
```bash
python src/evaluate.py
```
This outputs and saves the Hit@10 metric in `results/metrics.txt`.

### 3. Run the Dashboard
Launches the interactive user interface:
```bash
streamlit run app.py
```

---

## 📊 Evaluation Results

The model has been optimized with negative sampling and regularized feedforward layers to prevent overfitting on MovieLens-100K:

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **Test RMSE** | `0.4743` | Average deviation of normalized rating predictions |
| **Hit@10** | `0.6554` | Percentage of users whose last rated item was ranked in top 10 recommendations |

---

## 💡 Hybrid Blending & Explanation Rules

### Hybrid Scoring formula
$$\text{Final Score} = (1 - w) \times \text{NCF\_Normalized} + w \times \text{Genre\_Affinity}$$
* **NCF (Collaborative Filtering)**: Recommends movies based on patterns of users with similar taste.
* **Genre Affinity (Content-Based)**: Measures the alignment between a movie's genres and the user's top-rated genres.
* **Weight Slider ($w$)**: Dynamically tunes the influence of either source at runtime.

### Explanation Examples
* **Genre Match**:
  > *"🎯 Why recommended? Matches your interest in **Drama** and **Thriller**."*
* **Collaborative Filtering Fallback**:
  > *"🤝 Recommended because users with similar preferences liked this movie."*

---

## 📜 Acknowledgements
* GroupLens research group for hosting the [MovieLens Dataset](https://grouplens.org/datasets/movielens/).
* He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Tat-Seng, C. (2017). *Neural Collaborative Filtering*. WWW '17.
