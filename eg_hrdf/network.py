"""Density flow networks: FiLM MLP (legacy, branch=2) and the Perceiver family."""

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


class _PerceiverBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))

    def forward(self, latents: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        kv_n = self.cross_norm(kv) if kv is not None else None
        attn_out, _ = self.cross_attn(self.cross_norm(latents), kv_n, kv_n, need_weights=False)
        latents = latents + attn_out
        attn_out, _ = self.self_attn(self.self_norm(latents), self.self_norm(latents),
                                     self.self_norm(latents), need_weights=False)
        latents = latents + attn_out
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class PerceiverDensityFlowNet(nn.Module):
    """Perceiver density-flow head (model.md): latent tokens extract features via
    cross-attention to conditioning tokens (timestep, node embedding, optional
    z / text / hash context), then predict the cell's b^3-way child distribution.
    """

    def __init__(
        self,
        branch: int = 4,
        n_blocks: int = 2,
        dim: int = 256,
        n_latent: int = 4,
        n_heads: int = 4,
        time_dim: int = 32,
        ctx_dim: int = 0,
        z_dim: int = 0,
        text_dim: int = 0,
    ):
        super().__init__()
        self.branch = branch
        self.out_dim = branch ** 3
        self.dim = dim
        self.n_latent = n_latent
        self.ctx_dim = ctx_dim
        self.z_dim = z_dim
        self.text_dim = text_dim
        self.time_dim = time_dim

        self.t_mlp = nn.Sequential(nn.Linear(time_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.e_mlp = nn.Sequential(nn.Linear(5, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.z_mlp = nn.Sequential(nn.Linear(z_dim, dim), nn.SiLU()) if z_dim > 0 else None
        self.text_mlp = nn.Sequential(nn.Linear(text_dim, dim), nn.SiLU()) if text_dim > 0 else None
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, dim), nn.SiLU()) if ctx_dim > 0 else None

        n_kv = 2 + (1 if z_dim > 0 else 0) + (1 if text_dim > 0 else 0) + (1 if ctx_dim > 0 else 0)
        self.n_kv = n_kv

        self.latent_in = nn.Linear(self.out_dim + 5, dim)
        self.latent_seed = nn.Parameter(torch.randn(n_latent, dim) * 0.02)
        self.blocks = nn.ModuleList([_PerceiverBlock(dim, n_heads) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, self.out_dim)

        nn.init.constant_(self.head.weight, 0)
        nn.init.constant_(self.head.bias, 0)

    def forward(
        self,
        p_t: torch.Tensor,
        t: torch.Tensor,
        e_B: torch.Tensor,
        ctx: torch.Tensor = None,
        z: torch.Tensor = None,
        text: torch.Tensor = None,
    ) -> torch.Tensor:
        B = p_t.shape[0]
        device = p_t.device
        kv = [self.t_mlp(timestep_embedding(t, self.time_dim))[:, None],
              self.e_mlp(e_B)[:, None]]
        if self.z_dim > 0:
            zz = z if z is not None else torch.zeros(B, self.z_dim, device=device)
            kv.append(self.z_mlp(zz)[:, None])
        if self.text_dim > 0:
            tt = text if text is not None else torch.zeros(B, self.text_dim, device=device)
            kv.append(self.text_mlp(tt)[:, None])
        if self.ctx_dim > 0:
            cc = ctx if ctx is not None else torch.zeros(B, self.ctx_dim, device=device)
            kv.append(self.ctx_mlp(cc)[:, None])
        kv = torch.cat(kv, dim=1)

        seed = self.latent_seed[None].expand(B, -1, -1)
        latents = self.latent_in(torch.cat([p_t, e_B], dim=-1))[:, None] + seed
        for block in self.blocks:
            latents = block(latents, kv)
        return self.head(latents.mean(dim=1))
