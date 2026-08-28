from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .coordinates import encode_coords


@dataclass
class DepthLevel:
    depth: int
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
    return -(p * np.log(p + eps)).sum(axis=-1) / np.log(8.0)


def build_flat_hierarchy(
    points: np.ndarray,
    max_depth: int = 6,
    lo: float = -1.0,
    hi: float = 1.0,
    child_occ_tol: float = 0.0,
) -> FlatOctree:
    from .coordinates import cell_center, quantize_points

    n_points = len(points)
    assert n_points > 0
    leaf_q = quantize_points(points, max_depth, lo, hi)
    leaf_keys = encode_coords(leaf_q, max_depth)
    leaf_keys.sort()
    uniq_keys, leaf_counts = np.unique(leaf_keys, return_counts=True)
    leaf_coords = _decode(uniq_keys, max_depth)

    levels: Dict[int, DepthLevel] = {}
    child_keys = uniq_keys
    child_counts = leaf_counts.astype(np.float64)

    for depth in range(max_depth - 1, -1, -1):
        scale = 2 ** depth
        child_coords = _decode(child_keys, depth + 1)
        parent_coords = child_coords >> 1
        parent_keys = encode_coords(parent_coords, depth)
        uniq_parent, inverse = np.unique(parent_keys, return_inverse=True)
        n_parents = len(uniq_parent)
        parent_counts = np.zeros(n_parents, dtype=np.int64)
        np.add.at(parent_counts, inverse, child_counts.astype(np.int64))

        occupancy = np.zeros((n_parents, 8), dtype=np.float64)
        parent_index = np.searchsorted(uniq_parent, parent_keys)
        octants = (
            (child_coords[:, 0] & 1)
            | ((child_coords[:, 1] & 1) << 1)
            | ((child_coords[:, 2] & 1) << 2)
        )
        np.add.at(occupancy, (parent_index, octants), child_counts)
        occupancy /= parent_counts[:, None]

        mass = parent_counts / n_points
        centers = cell_center(_decode(uniq_parent, depth), depth, lo, hi)
        levels[depth] = DepthLevel(
            depth=depth,
            coords=_decode(uniq_parent, depth),
            counts=parent_counts,
            mass=mass,
            occupancy=occupancy,
            entropy=_entropy(occupancy),
            occupancy_var=((occupancy - 0.125) ** 2).sum(axis=1),
            child_mask=(occupancy > child_occ_tol).astype(np.uint8),
            centers=centers,
        )
        child_keys = uniq_parent
        child_counts = parent_counts.astype(np.float64)

    return FlatOctree(
        n_points=n_points,
        max_depth=max_depth,
        levels=levels,
        leaf_coords=leaf_coords,
        leaf_counts=leaf_counts,
    )


def _decode(keys: np.ndarray, depth: int) -> np.ndarray:
    from .coordinates import decode_coords

    return decode_coords(np.asarray(keys, dtype=np.int64), depth)


def occupancy_to_records(level: DepthLevel) -> List[dict]:
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
