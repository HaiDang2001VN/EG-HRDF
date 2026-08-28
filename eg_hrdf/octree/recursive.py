"""Adaptive octree representation of point clouds (m_B, p_B, e_B fields)."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

OCTANT_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int64,
)


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    pts = (points - lo) / span
    return np.clip(pts, 0.0, 1.0 - 1e-7), lo, span


def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(p * np.log(p + eps), axis=-1)


def normalized_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return shannon_entropy(p, eps) / np.log(8.0)


@dataclass
class Octree:
    points: np.ndarray
    max_depth: int
    parent: np.ndarray
    depth: np.ndarray
    cell: np.ndarray
    mass: np.ndarray
    p: np.ndarray
    centroid: np.ndarray
    children: np.ndarray
    origin: np.ndarray
    span: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.depth)

    def internal_mask(self) -> np.ndarray:
        return (self.children >= 0).any(axis=1)

    def nodes_at_depth(self, d: int) -> np.ndarray:
        return np.nonzero(self.depth == d)[0]

    def cut_at_depth(self, d: int) -> np.ndarray:
        d = min(d, self.max_depth)
        cut = []
        stack = [0]
        internal = self.internal_mask()
        while stack:
            node = stack.pop()
            if self.depth[node] >= d or not internal[node]:
                cut.append(node)
            else:
                children = self.children[node]
                stack.extend(children[children >= 0].tolist())
        return np.array(sorted(cut), dtype=np.int64)

    def check_mass_conservation(self, tol: float = 1e-6) -> bool:
        root_mass = self.mass[0]
        cut = self.cut_at_depth(self.max_depth)
        return bool(abs(self.mass[cut].sum() - root_mass) < tol)

    def block_embeddings(self) -> np.ndarray:
        depth_frac = self.depth.astype(np.float64) / max(self.max_depth, 1)
        return np.concatenate(
            [self.centroid, depth_frac[:, None], self.mass[:, None]], axis=1
        )


def build_octree(
    points: np.ndarray,
    max_depth: int = 6,
    normalize: bool = True,
    origin: Optional[np.ndarray] = None,
    span: Optional[np.ndarray] = None,
    min_points: int = 1,
) -> Octree:
    if normalize:
        pts, origin, span = normalize_points(points)
    else:
        pts = np.clip(points, 0.0, 1.0 - 1e-7)
        if origin is None:
            origin = np.zeros(3)
        if span is None:
            span = np.ones(3)
    n_points = len(pts)
    assert n_points > 0

    parent_list, depth_list, cell_list = [], [], []
    mass_list, p_list, centroid_list, children_list = [], [], [], []

    root_p = np.zeros(8, dtype=np.float64)

    def _mass_of(idx: np.ndarray) -> float:
        return len(idx) / n_points

    def _rec(parent: int, depth: int, cell: np.ndarray, idx: np.ndarray) -> int:
        node = len(parent_list)
        parent_list.append(parent)
        depth_list.append(depth)
        cell_list.append(cell.copy())
        mass_list.append(_mass_of(idx))
        p_list.append(np.full(8, np.nan))
        centroid_list.append(pts[idx].mean(axis=0))
        children_list.append(np.full(8, -1, dtype=np.int64))

        if depth == max_depth or len(idx) <= min_points:
            return node

        scale = 2 ** (depth + 1)
        child_cells = (pts[idx] * scale).astype(np.int64) - 2 * cell
        child_p = np.zeros(8, dtype=np.float64)
        for octant in range(8):
            mask = np.all(child_cells == OCTANT_OFFSETS[octant], axis=1)
            n_oct = int(mask.sum())
            child_p[octant] = n_oct / len(idx)
            if n_oct == 0:
                continue
            child_cell = 2 * cell + OCTANT_OFFSETS[octant]
            child_node = _rec(node, depth + 1, child_cell, idx[mask])
            children_list[node][octant] = child_node
        p_list[node] = child_p
        return node

    _rec(-1, 0, np.zeros(3, dtype=np.int64), np.arange(n_points))

    return Octree(
        points=pts,
        max_depth=max_depth,
        parent=np.array(parent_list, dtype=np.int64),
        depth=np.array(depth_list, dtype=np.int64),
        cell=np.array(cell_list, dtype=np.int64),
        mass=np.array(mass_list, dtype=np.float64),
        p=np.array(p_list, dtype=np.float64),
        centroid=np.array(centroid_list, dtype=np.float64),
        children=np.array(children_list, dtype=np.int64),
        origin=np.asarray(origin, dtype=np.float64),
        span=np.asarray(span, dtype=np.float64),
    )


def denormalize_points(points: np.ndarray, origin: np.ndarray, span: np.ndarray) -> np.ndarray:
    return points * span[None] + origin[None]
