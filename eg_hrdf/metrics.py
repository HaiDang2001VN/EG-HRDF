"""Density-aware Chamfer Distance (DCD) and helpers, pure PyTorch."""

import torch


def _pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(-1).unsqueeze(1)
    y2 = (y * y).sum(-1).unsqueeze(0)
    dist = x2 + y2 - 2.0 * torch.mm(x, y.t())
    return dist.clamp(min=0)


def _ncd(x: torch.Tensor, y: torch.Tensor) -> tuple:
    dist = _pairwise_dist(x, y).sqrt()
    dist = dist / (dist.max() + 1e-8)
    return dist.min(1)[0], dist.min(0)[0]


def _density_weights(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    n = x.shape[0]
    k = min(k, n - 1) if n > 1 else 1
    dist = _pairwise_dist(x, x).sqrt()
    vals, _ = dist.topk(k + 1, dim=1, largest=False)
    rho = vals[:, 1:].mean(1)
    w = torch.exp(-rho)
    return w / (w.sum() + 1e-8)


def density_aware_chamfer(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int = 4,
) -> torch.Tensor:
    d1, d2 = _ncd(x, y)
    w_x = _density_weights(x, k)
    w_y = _density_weights(y, k)
    return 0.5 * ((w_x * d1).sum() + (w_y * d2).sum())


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple:
    dist = _pairwise_dist(x, y)
    return dist.min(1)[0], dist.min(0)[0]
