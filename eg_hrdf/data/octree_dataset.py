"""Octree-node datasets for EG-HRDF training."""

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from ..octree import Octree, build_octree


class OctreeNodeDataset(Dataset):
    def __init__(
        self,
        octrees: List[Octree],
        blocks_per_shape: int = 256,
        alpha: float = 0.5,
        lam: float = 0.5,
        eps: float = 1e-8,
    ):
        self.octrees = octrees
        self.blocks_per_shape = blocks_per_shape
        self.alpha = alpha
        self.lam = lam
        self.eps = eps
        self._cache = [self._preprocess(tree) for tree in octrees]

    @staticmethod
    def _preprocess(tree: Octree):
        internal = tree.internal_mask()
        idx = np.nonzero(internal)[0]
        p1 = tree.p[idx]
        mass = tree.mass[idx]
        e = np.concatenate(
            [
                tree.centroid[idx],
                (tree.depth[idx] / max(tree.max_depth, 1))[:, None],
                mass[:, None],
            ],
            axis=1,
        )
        p_mass = np.clip(mass, 1e-8, None) ** 0.5
        p_mass = p_mass / p_mass.sum()
        p_uniform = np.full(len(idx), 1.0 / max(len(idx), 1))
        return {
            "idx": idx,
            "p1": torch.tensor(p1, dtype=torch.float32),
            "mass": torch.tensor(mass, dtype=torch.float32),
            "e": torch.tensor(e, dtype=torch.float32),
            "prob": torch.tensor(0.5 * p_mass + 0.5 * p_uniform, dtype=torch.float32),
            "origin": tree.origin,
            "span": tree.span,
        }

    def __len__(self) -> int:
        return len(self.octrees)

    def __getitem__(self, idx: int):
        item = self._cache[idx]
        k = min(self.blocks_per_shape, item["p1"].shape[0])
        sampled = torch.multinomial(item["prob"], k, replacement=True)
        return {
            "p1": item["p1"][sampled],
            "mass": item["mass"][sampled],
            "e": item["e"][sampled],
            "shape_idx": torch.full((k,), idx, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        out = {}
        for key in ("p1", "mass", "e", "shape_idx"):
            out[key] = torch.cat([b[key] for b in batch], dim=0)
        return out


def build_octree_dataset_from_points(
    point_clouds: List[np.ndarray],
    max_depth: int = 6,
    blocks_per_shape: int = 256,
    alpha: float = 0.5,
    lam: float = 0.5,
) -> OctreeNodeDataset:
    octrees = [build_octree(pc, max_depth=max_depth) for pc in point_clouds]
    return OctreeNodeDataset(octrees, blocks_per_shape=blocks_per_shape, alpha=alpha, lam=lam)
