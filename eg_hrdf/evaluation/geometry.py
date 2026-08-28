"""Generative point-cloud metrics: COV / MMD / 1-NNA (CD-based) + DCD, pure PyTorch."""

from typing import Dict

import torch

from ..metrics import _pairwise_dist, density_aware_chamfer


def _cd_matrix(gen: torch.Tensor, ref: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Chamfer distance matrix between two sets of point clouds.

    gen: (G, N, 3), ref: (R, N, 3) -> (G, R), symmetric CD = 0.5*(d_xy + d_yx).
    """
    G, R = gen.shape[0], ref.shape[0]
    out = torch.zeros(G, R)
    for i in range(0, G, chunk):
        g = gen[i : i + chunk]
        for j in range(0, R, chunk):
            r = ref[j : j + chunk]
            n_g, n_r = g.shape[1], r.shape[1]
            d = _pairwise_dist(g.reshape(-1, 3), r.reshape(-1, 3))
            d = d.reshape(g.shape[0], n_g, r.shape[0], n_r)
            dxy = d.min(dim=3)[0].mean(dim=1)
            dyx = d.min(dim=1)[0].mean(dim=2)
            out[i : i + chunk, j : j + chunk] = 0.5 * (dxy + dyx)
    return out


def cov_mmd_1nna(gen: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """COV / MMD / 1-NNA following the standard point-cloud generative protocol."""
    G, R = gen.shape[0], ref.shape[0]
    d_gr = _cd_matrix(gen, ref)
    d_gg = _cd_matrix(gen, gen)
    d_rr = _cd_matrix(ref, ref)

    idx = d_gr.argmin(dim=0)
    cov = len(set(idx.tolist())) / G

    mmd = d_gr.min(dim=1)[0].mean().item()

    M = torch.zeros(G + R, G + R)
    M[:G, :G] = d_gg
    M[:G, G:] = d_gr
    M[G:, :G] = d_gr.t()
    M[G:, G:] = d_rr
    M.fill_diagonal_(float("inf"))
    nn_idx = M.argmin(dim=1)
    true_gen = torch.cat([torch.ones(G), torch.zeros(R)]).bool()
    pred_gen = true_gen[nn_idx]
    acc = (pred_gen == true_gen).float().mean().item()
    return {"COV": cov, "MMD-CD": mmd, "1-NNA-CD": acc}


def mmd_dcd(gen: torch.Tensor, ref: torch.Tensor, k: int = 4, max_pairs: int = 64) -> float:
    """MMD with density-aware chamfer on a bounded number of pairs."""
    G, R = gen.shape[0], ref.shape[0]
    n = min(min(G, R), max_pairs)
    vals = []
    for _ in range(n):
        g = gen[int(torch.randint(0, G, (1,)))]
        r = ref[int(torch.randint(0, R, (1,)))]
        vals.append(density_aware_chamfer(g, r, k=k).item())
    return sum(vals) / max(len(vals), 1)


def jsd_between_point_cloud_sets(gen: torch.Tensor, ref: torch.Tensor, resolution: int = 28, lo: float = -1.0, hi: float = 1.0) -> float:
    """JSD between occupancy distributions over a voxel grid (evaluation convention)."""
    import numpy as np

    def occupancy(pcs):
        H = np.zeros((resolution, resolution, resolution))
        for pc in pcs:
            q = np.clip(((pc - lo) / (hi - lo) * resolution).astype(np.int64), 0, resolution - 1)
            H[q[:, 0], q[:, 1], q[:, 2]] += 1
        H /= H.sum()
        return H.reshape(-1)

    def jsdiv(p, q):
        p = np.clip(p, 1e-12, None)
        q = np.clip(q, 1e-12, None)
        m = 0.5 * (p + q)
        return 0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum()

    p = occupancy([g.numpy() for g in gen])
    q = occupancy([r.numpy() for r in ref])
    return float(jsdiv(p, q))
