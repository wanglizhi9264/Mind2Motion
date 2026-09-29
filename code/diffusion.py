"""Text-conditioned diffusion model used to generate HumanML3D motion vectors."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_dim // 2
        exponents = torch.arange(
            half_dim, device=timestep.device, dtype=torch.float32
        ) / half_dim
        frequencies = torch.exp(-math.log(10000.0) * exponents)
        angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = torch.cat(
                [
                    embedding,
                    torch.zeros(
                        embedding.shape[0],
                        self.hidden_dim - embedding.shape[1],
                        device=timestep.device,
                    ),
                ],
                dim=-1,
            )
        return self.mlp(embedding)


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))

    def forward(self, tokens: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mlp(modulation).chunk(2, dim=-1)
        return self.layernorm(tokens) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MotionTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ada_ln_1 = AdaLayerNorm(hidden_dim)
        self.ada_ln_2 = AdaLayerNorm(hidden_dim)
        self.ffn_ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        time_embedding: torch.Tensor,
        text_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        padding_mask = mask == 0 if mask is not None else None
        normalized = self.ada_ln_1(tokens, time_embedding)
        attended, _ = self.self_attn(
            normalized, normalized, normalized, key_padding_mask=padding_mask
        )
        tokens = tokens + self.dropout(attended)

        normalized = self.ada_ln_2(tokens, time_embedding)
        attended, _ = self.cross_attn(
            normalized, text_embeddings, text_embeddings
        )
        tokens = tokens + self.dropout(attended)
        return tokens + self.dropout(self.ffn(self.ffn_ln(tokens)))


class MotionDiffusionTransformer(nn.Module):
    """Predict clean 263-D motion from noisy motion and four condition tokens."""

    def __init__(
        self,
        input_feats: int = 263,
        max_seq_len: int = 196,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        text_latent_dim: int = 512,
        dropout: float = 0.2,
        pred_mode: str = "x",
    ) -> None:
        super().__init__()
        if pred_mode not in ("x", "eps"):
            raise ValueError(f"Unsupported pred_mode: {pred_mode}")
        self.input_feats = input_feats
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.pred_mode = pred_mode

        self.input_proj = nn.Linear(input_feats, hidden_dim)
        self.text_proj = nn.Linear(text_latent_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MotionTransformerBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_feats)
        self.self_cond_proj: Optional[nn.Linear] = None
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        for layer in (self.input_proj, self.text_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for module in self.modules():
            if isinstance(module, AdaLayerNorm):
                nn.init.zeros_(module.mlp[-1].weight)
                nn.init.zeros_(module.mlp[-1].bias)

    def forward(
        self,
        motion: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, sequence_length, feature_dim = motion.shape
        if feature_dim != self.input_feats:
            raise ValueError(f"Expected {self.input_feats} motion features, got {feature_dim}")
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds {self.max_seq_len}"
            )
        if self.self_cond_proj is not None and self_cond is not None:
            motion = motion + self.self_cond_proj(self_cond)

        tokens = self.input_proj(motion) + self.pos_embed[:, :sequence_length]
        text_context = self.text_proj(text_embeds)
        time_embedding = self.time_embedder(timestep)
        for block in self.blocks:
            tokens = block(tokens, time_embedding, text_context, mask)
        return self.out_proj(self.out_ln(tokens))


def linear_beta_schedule(
    steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)


class DDIMSampler(nn.Module):
    """Minimal x-prediction DDIM sampler matching the released checkpoint."""

    def __init__(self, betas: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if betas is None:
            betas = linear_beta_schedule()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def _timesteps(self, ddim_steps: int) -> torch.Tensor:
        total_steps = len(self.betas)
        ddim_steps = max(1, min(int(ddim_steps), total_steps))
        if ddim_steps == 1:
            steps = np.asarray([total_steps - 1], dtype=np.int64)
        else:
            stride = max(1, total_steps // ddim_steps)
            steps = np.arange(0, total_steps, stride, dtype=np.int64)[:ddim_steps]
            steps[-1] = total_steps - 1
        return torch.from_numpy(steps[::-1].copy()).long().to(self.betas.device)

    @torch.no_grad()
    def sample(
        self,
        model: MotionDiffusionTransformer,
        text_embeddings: torch.Tensor,
        frames: int = 196,
        ddim_steps: int = 50,
        eta: float = 0.0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = text_embeddings.shape[0]
        motion = torch.randn(
            batch_size,
            frames,
            model.input_feats,
            device=text_embeddings.device,
        )
        timesteps = self._timesteps(ddim_steps)
        for index, timestep in enumerate(timesteps):
            current = torch.full(
                (batch_size,), int(timestep), device=motion.device, dtype=torch.long
            )
            prediction = model(motion, current, text_embeddings, mask=mask)
            if index == len(timesteps) - 1:
                motion = prediction
                continue

            previous = torch.full_like(current, int(timesteps[index + 1]))
            alpha = self.alphas_cumprod[current].view(-1, 1, 1)
            alpha_previous = self.alphas_cumprod[previous].view(-1, 1, 1)
            noise_prediction = (
                motion - torch.sqrt(alpha) * prediction
            ) / (torch.sqrt(1.0 - alpha) + 1e-8)
            sigma = eta * torch.sqrt(
                (1.0 - alpha_previous)
                / (1.0 - alpha)
                * (1.0 - alpha / alpha_previous)
            )
            noise = torch.randn_like(motion) if eta > 0.0 else 0.0
            motion = (
                torch.sqrt(alpha_previous) * prediction
                + torch.sqrt(torch.clamp(1.0 - alpha_previous - sigma**2, min=0.0))
                * noise_prediction
                + sigma * noise
            )
        return motion
