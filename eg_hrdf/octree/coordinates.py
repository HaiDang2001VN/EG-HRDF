import numpy as np


def quantize_points(points: np.ndarray, depth: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    scale = 2 ** depth
    q = np.floor((points - lo) / (hi - lo) * scale).astype(np.int64)
    return np.clip(q, 0, scale - 1)


def cell_center(coords: np.ndarray, depth: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    scale = 2 ** depth
    return lo + (coords + 0.5) / scale * (hi - lo)


def encode_coords(coords: np.ndarray, depth: int) -> np.ndarray:
    scale = 2 ** depth
    return (coords[:, 0] * scale + coords[:, 1]) * scale + coords[:, 2]


def decode_coords(keys: np.ndarray, depth: int) -> np.ndarray:
    scale = 2 ** depth
    x = keys // (scale * scale)
    rem = keys - x * scale * scale
    y = rem // scale
    z = rem - y * scale
    return np.stack([x, y, z], axis=1)
