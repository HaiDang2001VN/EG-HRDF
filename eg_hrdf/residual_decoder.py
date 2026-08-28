"""Local residual point decoder for terminal blocks (data.md 24)."""

import torch
import torch.nn as nn


class LocalResidualDecoder(nn.Module):
    def __init__(self, z_dim: int = 0, hidden: int = 128, n_local: int = 16):
        super().__init__()
        self.n_local = n_local
        self.z_dim = z_dim
        self.point_embed = nn.Embedding(n_local, 16)
        input_dim = 5 + 8 + 16 + z_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
            nn.Tanh(),
        )

    def forward(
        self,
        e_B: torch.Tensor,
        p_B: torch.Tensor,
        center: torch.Tensor,
        extent: torch.Tensor,
        z: torch.Tensor = None,
    ) -> torch.Tensor:
        """e_B (B,5), p_B (B,8), center (B,3), extent (B,3) in [-1,1] domain.

        Returns (B, n_local, 3) points: center + extent/2 * tanh(mlp(...)).
        """
        B = e_B.shape[0]
        idx = torch.arange(self.n_local, device=e_B.device)
        pe = self.point_embed(idx)[None].expand(B, -1, -1)
        e = e_B[:, None, :].expand(B, self.n_local, -1)
        p = p_B[:, None, :].expand(B, self.n_local, -1)
        parts = [e, p, pe]
        if self.z_dim > 0:
            zz = z[:, None, :].expand(B, self.n_local, -1) if z is not None \
                else torch.zeros(B, self.n_local, self.z_dim, device=e_B.device)
            parts.insert(2, zz)
        x = torch.cat(parts, dim=-1)
        r = self.mlp(x)
        return center[:, None, :] + 0.5 * extent[:, None, :] * r
