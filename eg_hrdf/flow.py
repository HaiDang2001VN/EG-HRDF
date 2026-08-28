"""Simplex-space rectified density flow: training objective and helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

UNIFORM_P0 = 1.0 / 8.0


def uniform_p0(batch: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch, 8), UNIFORM_P0, device=device, dtype=torch.float32)


class SimplexDensityFlowMatcher:
    def __init__(self, gamma: float = 0.5, alpha: float = 0.5, lam: float = 0.5, eps: float = 1e-8):
        self.gamma = gamma
        self.alpha = alpha
        self.lam = lam
        self.eps = eps

    def interpolate(self, p1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p0 = uniform_p0(p1.shape[0], p1.device)
        t = t[:, None]
        return (1.0 - t) * p0 + t * p1

    def sample_t(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch, device=device, dtype=torch.float32)

    def block_sampling_prob(self, mass: torch.Tensor) -> torch.Tensor:
        p_mass = mass.clamp_min(self.eps) ** self.alpha
        p_mass = p_mass / p_mass.sum()
        p_uniform = torch.full_like(p_mass, 1.0 / mass.shape[0])
        return self.lam * p_mass + (1.0 - self.lam) * p_uniform

    def endpoint_loss(
        self,
        logits: torch.Tensor,
        p1: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        log_p_hat = F.log_softmax(logits, dim=-1)
        ce = -(p1 * log_p_hat).sum(dim=-1)
        w = mass.clamp_min(self.eps) ** self.gamma
        w = w / w.sum()
        return (w * ce).sum()

    def training_loss(self, model, p1: torch.Tensor, e_B: torch.Tensor, mass: torch.Tensor,
                      ctx: torch.Tensor = None, z: torch.Tensor = None) -> torch.Tensor:
        t = self.sample_t(p1.shape[0], p1.device)
        p_t = self.interpolate(p1, t)
        logits = model(p_t, t, e_B, ctx=ctx, z=z)
        return self.endpoint_loss(logits, p1, mass)

    @staticmethod
    def velocity(logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p1_hat = F.softmax(logits, dim=-1)
        p0 = uniform_p0(p1_hat.shape[0], p1_hat.device)
        return p1_hat - p0
