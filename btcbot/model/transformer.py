"""A small pre-norm transformer encoder that maps a window of bar features to one logit for P(up)."""
from __future__ import annotations

import torch
from torch import nn


class DirectionTransformer(nn.Module):
    def __init__(self, n_features: int, seq_len: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> logits (batch,)"""
        h = self.input_proj(x) + self.pos_emb[:, : x.shape[1]]
        cls = self.cls.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0])).squeeze(-1)
