import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from .modules import BiMatchingNet, TreeGateBranchingNet

class Critic(nn.Module):
    """Critic network for PPO that estimates state values."""

    def __init__(self, var_dim, node_dim, mip_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.var_embedding = nn.Sequential(
            nn.LayerNorm(var_dim),
            nn.Linear(var_dim, hidden_dim),
        )
        self.tree_embedding = nn.Sequential(
            nn.LayerNorm(node_dim + mip_dim),
            nn.Linear(node_dim + mip_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.tree_refinement = BiMatchingNet(hidden_dim)

        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.value_head = TreeGateBranchingNet(
            branch_size=hidden_dim,
            tree_state_size=hidden_dim,
            dim_reduce_factor=2,
            infimum=1,
            norm='layer',
            depth=2,
            hidden_size=hidden_dim,
        )

    def forward(self, cands_state_mat, node_state, mip_state, padding_mask=None):
        batch_size, num_candidates, _ = cands_state_mat.shape

        var_features = self.var_embedding(cands_state_mat)
        var_features = var_features.transpose(0, 1)  # [L, B, H]

        if padding_mask is not None:
            src_key_padding_mask = padding_mask
        else:
            src_key_padding_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool, device=cands_state_mat.device)

        transformed_features = self.transformer(var_features, src_key_padding_mask=src_key_padding_mask)
        transformed_features = transformed_features.transpose(0, 1)  # [B, L, H]

        tree_state = torch.cat([node_state, mip_state], dim=-1)
        tree_features = self.tree_embedding(tree_state)  # [B, H]

        refined_features = self.tree_refinement(tree_features, transformed_features, src_key_padding_mask)

        valid_mask = (~src_key_padding_mask).unsqueeze(-1)  # [B, L, 1]
        masked_sum = (refined_features * valid_mask).sum(dim=1)
        valid_count = valid_mask.sum(dim=1).clamp(min=1)
        pooled_features = masked_sum / valid_count

        combined = torch.cat([pooled_features, tree_features], dim=-1)
        aggregated = self.aggregation(combined)

        value = self.value_head(aggregated, tree_features).unsqueeze(-1)  # [B, 1]
        return value
