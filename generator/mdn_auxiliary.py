"""Proposal-conditioned auxiliary MDN model for offline representation learning."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import Tensor, nn


PathLike = Union[str, Path]


class MDNAuxiliaryModel(nn.Module):
    """Learn proposal-conditioned accept and motive-return signals.
    Inputs: `context`: shape `(context_dim,)` or `(N, context_dim)` and `skill_id`: shape `()` or `(N,)`
    Outputs: `gate_logits`: shape `()` or `(N,)` and `q_hat`: shape `(num_motives,)` or `(N, num_motives)`
    """

    def __init__(
        self,
        context_dim: int = 14,
        num_skills: int = 128,
        skill_embedding_dim: int = 8,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        num_motives: int = 2,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError(f"Expected num_hidden_layers >= 1, got {num_hidden_layers}")
        if num_skills <= 0:
            raise ValueError(f"Expected num_skills > 0, got {num_skills}")

        self.context_dim = context_dim
        self.num_skills = num_skills
        self.skill_embedding_dim = skill_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_motives = num_motives

        self.skill_embedding = nn.Embedding(num_skills, skill_embedding_dim)
        input_dim = context_dim + skill_embedding_dim
        trunk_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            trunk_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )

        self.trunk = nn.Sequential(*trunk_layers)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.motive_head = nn.Linear(hidden_dim, num_motives)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.skill_embedding.weight, mean=0.0, std=0.02)

    def forward(self, context: Tensor, skill_id: Tensor) -> tuple[Tensor, Tensor]:
        if context.ndim not in (1, 2):
            raise ValueError(
                f"Expected context with shape ({self.context_dim},) or (N, {self.context_dim}), got {tuple(context.shape)}"
            )

        is_single_input = context.ndim == 1
        if is_single_input:
            if context.shape[0] != self.context_dim:
                raise ValueError(
                    f"Expected single context shape ({self.context_dim},), got {tuple(context.shape)}"
                )
            context = context.unsqueeze(0)
            if skill_id.ndim != 0:
                raise ValueError(f"Expected scalar skill_id for single context, got shape {tuple(skill_id.shape)}")
            skill_id = skill_id.unsqueeze(0)
        else:
            if context.shape[1] != self.context_dim:
                raise ValueError(
                    f"Expected batched context shape (N, {self.context_dim}), got {tuple(context.shape)}"
                )
            if skill_id.ndim != 1:
                raise ValueError(f"Expected batched skill_id shape (N,), got shape {tuple(skill_id.shape)}")
            if skill_id.shape[0] != context.shape[0]:
                raise ValueError("skill_id batch size must match context batch size")

        embedded_skill = self.skill_embedding(skill_id.long())
        features = torch.cat([context, embedded_skill], dim=-1)
        hidden = self.trunk(features)
        gate_logits = self.gate_head(hidden).squeeze(-1)
        q_hat = self.motive_head(hidden)

        if is_single_input:
            gate_logits = gate_logits.squeeze(0)
            q_hat = q_hat.squeeze(0)

        return gate_logits, q_hat

    def save(self, path: PathLike) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: PathLike, map_location: str | torch.device = "cpu") -> None:
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
