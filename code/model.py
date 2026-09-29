"""Mind2Motion EEG encoder.

Input:  EEG trials shaped ``(batch, channels, time)``.
Output: a normalized text-space embedding and motion-class logits.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(tokens), dim=1)
        return torch.sum(weights * tokens, dim=1)


class EEGEncoder(nn.Module):
    """Multi-scale temporal-spatial EEG encoder with gated context."""

    def __init__(
        self,
        channels: int = 32,
        embedding_dim: int = 512,
        num_classes: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        branch_filters = 16
        temporal_filters = branch_filters * 3
        spatial_filters = temporal_filters * 2
        model_dim = 128

        self.temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        1,
                        branch_filters,
                        kernel_size=(1, kernel),
                        padding=(0, kernel // 2),
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_filters),
                )
                for kernel in (15, 31, 63)
            ]
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
        )
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(
                spatial_filters,
                spatial_filters,
                kernel_size=15,
                padding=7,
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv1d(spatial_filters, model_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.transformer_norm = nn.LayerNorm(model_dim)
        self.transformer_gate = nn.Parameter(torch.tensor(-2.0))

        self.text_pool = AttentionPool(model_dim)
        self.text_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )
        self.classification_head = nn.Sequential(
            nn.LayerNorm(model_dim * 3),
            nn.Dropout(0.2),
            nn.Linear(model_dim * 3, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, num_classes),
        )

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = torch.cat(
            [branch(eeg.unsqueeze(1)) for branch in self.temporal_branches], dim=1
        )
        spatial = self.spatial(temporal).squeeze(2)
        local = self.temporal_refine(spatial).transpose(1, 2)
        context = self.transformer_norm(self.transformer(local))
        fused = local + torch.sigmoid(self.transformer_gate) * context

        embedding = F.normalize(self.text_head(self.text_pool(fused)), dim=1)
        class_features = torch.cat(
            [local.mean(dim=1), local.amax(dim=1), fused.mean(dim=1)], dim=1
        )
        logits = self.classification_head(class_features)
        return embedding, logits
