import numpy as np

from .builder import DepthLevel


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(p * np.log(p + eps)).sum(axis=-1) / np.log(8.0)


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
