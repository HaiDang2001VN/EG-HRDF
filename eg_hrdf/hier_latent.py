"""Hierarchical stochastic latents: z_{B_i} = g_phi(z_B, eps_i, e_{B_i})."""

import torch
import torch.nn as nn


class HierarchicalLatentGenerator(nn.Module):
    def __init__(self, z_dim: int = 16, embed_dim: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.g = nn.Sequential(
            nn.Linear(2 * z_dim + embed_dim, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
        )

    def root(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch, self.z_dim, device=device)

    def expand(self, z_parent: torch.Tensor, e_child: torch.Tensor) -> torch.Tensor:
        eps = torch.randn(z_parent.shape[0], self.z_dim, device=z_parent.device)
        if e_child.shape[-1] != self.g[0].in_features - 2 * self.z_dim:
            e_child = e_child[..., : self.g[0].in_features - 2 * self.z_dim]
        return torch.tanh(self.g(torch.cat([z_parent, eps, e_child], dim=-1)))
