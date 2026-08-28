import numpy as np


def quantize_points(points: np.ndarray, depth: int, branch: int = 2, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    scale = branch ** depth
    q = np.floor((points - lo) / (hi - lo) * scale).astype(np.int64)
    return np.clip(q, 0, scale - 1)


def cell_center(coords: np.ndarray, depth: int, branch: int = 2, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    scale = branch ** depth
    return lo + (coords + 0.5) / scale * (hi - lo)


def encode_coords(coords: np.ndarray, depth: int, branch: int = 2) -> np.ndarray:
    scale = branch ** depth
    return (coords[:, 0] * scale + coords[:, 1]) * scale + coords[:, 2]


def decode_coords(keys: np.ndarray, depth: int, branch: int = 2) -> np.ndarray:
    scale = branch ** depth
    x = keys // (scale * scale)
    rem = keys - x * scale * scale
    y = rem // scale
    z = rem - y * scale
    return np.stack([x, y, z], axis=1)


def octant_of(child_cell: np.ndarray, parent_cell: np.ndarray, branch: int) -> np.ndarray:
    """Base-b octant index of child relative to parent: o = dx + b*dy + b^2*dz."""
    d = child_cell - branch * parent_cell
    return d[:, 0] + branch * d[:, 1] + branch * branch * d[:, 2]


def octant_offsets(branch: int) -> np.ndarray:
    b = branch
    offs = np.zeros((b ** 3, 3), dtype=np.int64)
    for o in range(b ** 3):
        offs[o] = [o % b, (o // b) % b, (o // (b * b)) % b]
    return offs


def n_children(branch: int) -> int:
    return branch ** 3
