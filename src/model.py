import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF).

    Combines two complementary branches:
      - GMF (Generalized Matrix Factorization): element-wise product
        of user and item embeddings — captures linear interactions.
      - MLP: concatenation of embeddings fed through hidden layers
        — captures non-linear interactions.

    Both branches are fused before the final prediction layer.
    """

    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()

        # ---------- GMF Branch ----------
        self.gmf_user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.gmf_item_embedding = nn.Embedding(
            num_items,
            embedding_dim
        )

        # ---------- MLP Branch ----------
        self.mlp_user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.mlp_item_embedding = nn.Embedding(
            num_items,
            embedding_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.35),
        )

        # ---------- Fusion ----------
        # GMF outputs embedding_dim features,
        # MLP outputs 64 features → fuse both
        self.prediction = nn.Linear(
            embedding_dim + 64,
            1
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user, item):

        # GMF branch
        gmf_user = self.gmf_user_embedding(user)
        gmf_item = self.gmf_item_embedding(item)
        gmf_out = gmf_user * gmf_item  # element-wise product

        # MLP branch
        mlp_user = self.mlp_user_embedding(user)
        mlp_item = self.mlp_item_embedding(item)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_out = self.mlp(mlp_input)

        # Fusion
        fusion = torch.cat([gmf_out, mlp_out], dim=1)

        return self.prediction(fusion).squeeze()