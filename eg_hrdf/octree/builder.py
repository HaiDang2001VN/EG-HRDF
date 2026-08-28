from dataclasses import dataclass
from typing import Dict

import numpy as np

from .coordinates import cell_center, decode_coords, encode_coords, n_children, octant_of, quantize_points


@dataclass
class DepthLevel:
    depth: int
    branch: int
    coords: np.ndarray
    counts: np.ndarray
    mass: np.ndarray
    occupancy: np.ndarray
    entropy: np.ndarray
    occupancy_var: np.ndarray
    child_mask: np.ndarray
    centers: np.ndarray


@dataclass
class FlatOctree:
    n_points: int
    max_depth: int
    branch: int
    levels: Dict[int, DepthLevel]
    leaf_coords: np.ndarray
    leaf_counts: np.ndarray

    def level(self, depth: int) -> DepthLevel:
        return self.levels[depth]

    def check_mass_conservation(self, tol: float = 1e-9) -> bool:
        for depth, level in self.levels.items():
            if abs(level.mass.sum() - 1.0) > tol:
                return False
        return abs(self.leaf_counts.sum() - self.n_points) == 0

    def node_count(self) -> int:
        return sum(len(l.coords) for l in self.levels.values())


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(p * np.log(p + eps)).sum(axis=-1) / np.log(p.shape[-1])


def build_flat_hierarchy(
    points: np.ndarray,
    max_depth: int,
    branch: int = 2,
    lo: float = -1.0,
    hi: float = 1.0,
    child_occ_tol: float = 0.0,
) -> FlatOctree:
    n_points = len(points)
    assert n_points > 0
    n_child = n_children(branch)

    leaf_q = quantize_points(points, max_depth, branch, lo, hi)
    leaf_keys = encode_coords(leaf_q, max_depth, branch)
    leaf_keys.sort()
    uniq_keys, leaf_counts = np.unique(leaf_keys, return_counts=True)
    leaf_coords = decode_coords(uniq_keys, max_depth, branch)

    levels: Dict[int, DepthLevel] = {}
    child_keys = uniq_keys
    child_counts = leaf_counts.astype(np.float64)

    for depth in range(max_depth - 1, -1, -1):
        child_coords = decode_coords(child_keys, depth + 1, branch)
        parent_coords = child_coords // branch
        parent_keys = encode_coords(parent_coords, depth, branch)
        uniq_parent, inverse = np.unique(parent_keys, return_inverse=True)
        n_parents = len(uniq_parent)
        parent_counts = np.zeros(n_parents, dtype=np.int64)
        np.add.at(parent_counts, inverse, child_counts.astype(np.int64))

        occupancy = np.zeros((n_parents, n_child), dtype=np.float64)
        parent_index = np.searchsorted(uniq_parent, parent_keys)
        octants = octant_of(child_coords, parent_coords, branch)
        np.add.at(occupancy, (parent_index, octants), child_counts)
        occupancy /= parent_counts[:, None]

        mass = parent_counts / n_points
        levels[depth] = DepthLevel(
            depth=depth,
            branch=branch,
            coords=decode_coords(uniq_parent, depth, branch),
            counts=parent_counts,
            mass=mass,
            occupancy=occupancy,
            entropy=_entropy(occupancy),
            occupancy_var=((occupancy - 1.0 / n_child) ** 2).sum(axis=1),
            child_mask=(occupancy > child_occ_tol).astype(np.uint8),
            centers=cell_center(decode_coords(uniq_parent, depth, branch), depth, branch, lo, hi),
        )
        child_keys = uniq_parent
        child_counts = parent_counts.astype(np.float64)

    return FlatOctree(
        n_points=n_points,
        max_depth=max_depth,
        branch=branch,
        levels=levels,
        leaf_coords=leaf_coords,
        leaf_counts=leaf_counts,
    )


def occupancy_to_records(level: DepthLevel):
    records = []
    for i in range(len(level.coords)):
        records.append(
            {
                "depth": level.depth,
                "node_coord": level.coords[i],
                "mass": float(level.mass[i]),
                "occupancy": level.occupancy[i],
                "entropy": float(level.entropy[i]),
                "occupancy_var": float(level.occupancy_var[i]),
                "child_mask": level.child_mask[i],
            }
        )
    return records


def level_to_arrays(level: DepthLevel) -> dict:
    return {
        "depth": level.depth,
        "coords": level.coords,
        "counts": level.counts,
        "mass": level.mass,
        "occupancy": level.occupancy,
        "entropy": level.entropy,
        "occupancy_var": level.occupancy_var,
        "child_mask": level.child_mask,
        "centers": level.centers,
    }
