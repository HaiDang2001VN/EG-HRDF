from typing import Optional

import numpy as np
import torch


class NodeSampler:
    def __init__(self, alpha: float = 0.5, lam: float = 0.5, eps: float = 1e-8, seed: Optional[int] = None):
        self.alpha = alpha
        self.lam = lam
        self.eps = eps
        self.rng = np.random.default_rng(seed)

    def probs(self, mass: np.ndarray) -> np.ndarray:
        p_mass = np.clip(mass, self.eps, None) ** self.alpha
        p_mass = p_mass / p_mass.sum()
        p_uniform = np.full(len(mass), 1.0 / len(mass))
        return self.lam * p_mass + (1.0 - self.lam) * p_uniform

    def sample(self, mass: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(mass))
        return self.rng.choice(len(mass), size=k, replace=False, p=self.probs(mass))

    def sample_batch(self, levels_masses: list, k_per_level: list):
        idx, lvl = [], []
        for li, (mass, k) in enumerate(zip(levels_masses, k_per_level)):
            sel = self.sample(mass, k)
            idx.append(sel)
            lvl.append(np.full(len(sel), li, dtype=np.int64))
        return np.concatenate(idx), np.concatenate(lvl)

    @staticmethod
    def to_torch(p1: np.ndarray, mass: np.ndarray, centers: np.ndarray, index: np.ndarray):
        return {
            "p1": torch.tensor(p1[index], dtype=torch.float32),
            "mass": torch.tensor(mass[index], dtype=torch.float32),
            "centers": torch.tensor(centers[index], dtype=torch.float32),
        }
