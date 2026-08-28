"""Hierarchical stochastic latents: z_{B_i} = g_phi(z_B, eps_i, e_{B_i})."""

import torch
import torch.nn as nn


class HierarchicalLatentGenerator(nn.Module):
    def __init__(self, z_dim: int = 16, embed_dim: int = 32, e_dim: int = 5):
        super().__init__()
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.e_proj = nn.Linear(e_dim, embed_dim)
        self.g = nn.Sequential(
            nn.Linear(2 * z_dim + embed_dim, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
        )

    def root(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch, self.z_dim, device=device)

    def expand(self, z_parent: torch.Tensor, e_child: torch.Tensor) -> torch.Tensor:
        if z_parent.dim() == 1:
            z_parent = z_parent.view(1, -1)
        if z_parent.shape[0] == 1 and e_child.shape[0] > 1:
            z_parent = z_parent.expand(e_child.shape[0], -1)
        eps = torch.randn(z_parent.shape[0], self.z_dim, device=z_parent.device)
        e = self.e_proj(e_child.to(z_parent.device, z_parent.dtype))
        return torch.tanh(self.g(torch.cat([z_parent, eps, e], dim=-1)))
