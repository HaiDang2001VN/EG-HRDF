"""Adaptive entropy-budgeted hierarchical generation (priority-queue expansion)."""

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .network import FiLMDensityFlowNet, PerceiverDensityFlowNet
from .octree.coordinates import octant_offsets


@dataclass
class SchedulerConfig:
    max_depth: int = 6
    branch: int = 2
    budget: Optional[int] = None
    score_mode: str = "entropy"
    tau_mass: float = 1e-3
    tau_refine: float = 0.0
    n_points: int = 2048
    point_source: str = "uniform"
    domain: tuple = (-1.0, 1.0)


@dataclass
class GenerationStats:
    evaluated: int = 0
    leaves: int = 0
    pruned: int = 0
    subdivided: int = 0
    rho: Optional[float] = None
    extra: Dict = field(default_factory=dict)


class AdaptiveDensityScheduler:
    def __init__(self, net, cfg: SchedulerConfig):
        self.net = net
        self.cfg = cfg
        self.n_children = cfg.branch ** 3
        self._offsets = octant_offsets(cfg.branch)

    def _node_embedding(self, cell: np.ndarray, depth: int, mass: float) -> torch.Tensor:
        d = self.cfg.branch ** depth
        center = (cell + 0.5) / d * 2 - 1
        depth_frac = depth / max(self.cfg.max_depth, 1)
        return torch.tensor(
            np.concatenate([center, [depth_frac, mass]]), dtype=torch.float32
        ).unsqueeze(0)

    @torch.no_grad()
    def evaluate_block(
        self,
        cell: np.ndarray,
        depth: int,
        mass: float,
        device: torch.device,
        z: Optional[torch.Tensor] = None,
        ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, float]:
        e = self._node_embedding(cell, depth, mass).to(device)
        p_t = torch.full((1, self.n_children), 1.0 / self.n_children, device=device, dtype=torch.float32)
        t = torch.zeros(1, device=device)
        logits = self.net(p_t, t, e, ctx=ctx, z=z)
        p1 = F.softmax(logits, dim=-1)[0].cpu().numpy()
        p = np.clip(p1, 1e-12, 1.0)
        h = float(-(p * np.log(p)).sum() / np.log(self.n_children))
        return p1, h

    def _score(self, h: float, mass: float) -> float:
        if self.cfg.score_mode == "entropy":
            return (1.0 - h) * mass
        if self.cfg.score_mode == "uniform":
            return mass
        raise ValueError(self.cfg.score_mode)

    @torch.no_grad()
    def generate(
        self,
        device: torch.device = torch.device("cpu"),
        z_root: Optional[torch.Tensor] = None,
        z_fn: Optional[Callable] = None,
        ctx_fn: Optional[Callable] = None,
        k_full: Optional[int] = None,
    ):
        cfg = self.cfg
        b = cfg.branch
        counter = itertools.count()
        heap = []
        root_cell = np.zeros(3, dtype=np.int64)
        p_root, h_root = self.evaluate_block(root_cell, 0, 1.0, device, z=z_root,
                                             ctx=ctx_fn(root_cell, 0) if ctx_fn else None)
        heapq.heappush(heap, (-self._score(h_root, 1.0), next(counter), 0, root_cell.copy(), 1.0, p_root, h_root, z_root))

        leaves = []
        evaluated = 0
        pruned = subdivided = 0

        while heap:
            neg_score, _, depth, cell, mass, p1, h, z = heapq.heappop(heap)
            score = -neg_score
            if mass < cfg.tau_mass:
                pruned += 1
                continue
            if evaluated >= (cfg.budget or float("inf")):
                leaves.append((cell, depth, mass))
                continue
            evaluated += 1

            if depth >= cfg.max_depth or score < cfg.tau_refine:
                leaves.append((cell, depth, mass))
                continue

            subdivided += 1
            for o in range(self.n_children):
                child_mass = mass * float(p1[o])
                if child_mass <= 0.0:
                    continue
                child_cell = b * cell + self._offsets[o]
                z_child = z
                if z_fn is not None:
                    e_child = self._node_embedding(child_cell, depth + 1, child_mass).to(device)
                    z_child = z_fn(z, e_child)
                ctx_child = ctx_fn(child_cell, depth + 1) if ctx_fn else None
                p_c, h_c = self.evaluate_block(child_cell, depth + 1, child_mass, device, z=z_child, ctx=ctx_child)
                heapq.heappush(heap, (-self._score(h_c, child_mass), next(counter), depth + 1, child_cell, child_mass, p_c, h_c, z_child))

        stats = GenerationStats(evaluated=evaluated, leaves=len(leaves), pruned=pruned, subdivided=subdivided)
        if k_full is not None and k_full > 0:
            stats.rho = evaluated / k_full

        if not leaves:
            return np.zeros((0, 3), dtype=np.float32), stats

        masses = np.array([m for _, _, m in leaves])
        total = masses.sum()
        if len(leaves) > cfg.n_points:
            keep = np.argsort(-masses)[: cfg.n_points]
            leaves = [leaves[i] for i in keep]
            masses = masses[keep]
            total = masses.sum()
        frac = masses / total * cfg.n_points
        counts = np.maximum(1, np.floor(frac)).astype(np.int64)
        rem = int(cfg.n_points - counts.sum())
        if rem > 0:
            order = np.argsort(-(frac - np.floor(frac)))
            counts[order[:rem]] += 1
        elif rem < 0:
            order = np.argsort(-counts)
            i = 0
            while rem < 0 and i < 100 * len(order):
                idx = order[i % len(order)]
                if counts[idx] > 1:
                    counts[idx] -= 1
                    rem += 1
                i += 1

        lo_d, hi_d = cfg.domain
        span = hi_d - lo_d
        pts = []
        rng = np.random
        for (cell, depth, mass), n_i in zip(leaves, counts):
            d = b ** depth
            lo = cell / d
            size = 1.0 / d
            u = rng.random((n_i, 3))
            pts.append(lo_d + (lo[None] + u * size) * span)
        points = np.concatenate(pts, axis=0).astype(np.float32)
        return points, stats
