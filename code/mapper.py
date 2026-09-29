"""Map one EEG embedding to the token sequence used by motion diffusion."""

from __future__ import annotations

import torch
from torch import nn


class EEGToTextTokenMapper(nn.Module):
    """Convert a 512-D EEG embedding into four 512-D condition tokens."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokens = tokens
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tokens * input_dim),
        )

    def forward(self, eeg_embedding: torch.Tensor) -> torch.Tensor:
        condition = self.net(eeg_embedding)
        return condition.reshape(eeg_embedding.shape[0], self.tokens, -1)
