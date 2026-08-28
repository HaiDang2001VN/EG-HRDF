"""Local rectified density flow network: f_theta(p_t, t, e_B, c_B, z_B) -> R^8."""

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class FiLMDensityFlowNet(nn.Module):
    def __init__(
        self,
        hidden_dims: tuple = (128, 256, 256),
        time_dim: int = 32,
        embed_dim: int = 32,
        ctx_dim: int = 0,
        z_dim: int = 0,
        text_dim: int = 0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        self.ctx_dim = ctx_dim
        self.z_dim = z_dim
        self.text_dim = text_dim

        cond_dim = time_dim + embed_dim + ctx_dim + z_dim + (32 if text_dim > 0 else 0)
        self.t_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.e_mlp = nn.Sequential(nn.Linear(5, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.z_mlp = nn.Sequential(nn.Linear(z_dim, z_dim), nn.SiLU()) if z_dim > 0 else None
        self.text_mlp = nn.Sequential(nn.Linear(text_dim, 32), nn.SiLU()) if text_dim > 0 else None

        dims = [8] + [h for h in hidden_dims]
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.films.append(nn.Linear(cond_dim, 2 * dims[i + 1]))
        self.out = nn.Linear(dims[-1], 8)

        self.blocks = len(hidden_dims)
        nn.init.constant_(self.out.weight, 0)
        nn.init.constant_(self.out.bias, 0)

    def forward(
        self,
        p_t: torch.Tensor,
        t: torch.Tensor,
        e_B: torch.Tensor,
        ctx: torch.Tensor = None,
        z: torch.Tensor = None,
        text: torch.Tensor = None,
    ) -> torch.Tensor:
        cond_parts = [self.t_mlp(timestep_embedding(t, self.time_dim)), self.e_mlp(e_B)]
        if self.ctx_dim > 0:
            cond_parts.append(ctx if ctx is not None else torch.zeros(p_t.shape[0], self.ctx_dim, device=p_t.device))
        if self.z_dim > 0 and self.z_mlp is not None:
            cond_parts.append(self.z_mlp(z) if z is not None else torch.zeros(p_t.shape[0], self.z_dim, device=p_t.device))
        if self.text_dim > 0 and self.text_mlp is not None:
            cond_parts.append(self.text_mlp(text) if text is not None else torch.zeros(p_t.shape[0], 32, device=p_t.device))
        cond = torch.cat(cond_parts, dim=-1)

        h = p_t
        for i in range(self.blocks):
            gamma, beta = self.films[i](cond).chunk(2, dim=-1)
            h = self.layers[i](h)
            h = h * (1.0 + gamma) + beta
            h = nn.functional.silu(h)
        return self.out(h)
