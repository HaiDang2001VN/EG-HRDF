import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf import LocalResidualDecoder, density_aware_chamfer
from eg_hrdf.evaluation import cov_mmd_1nna, jsd_between_point_cloud_sets, mmd_dcd


def _sphere(n=256, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    z = 2 * rng.random(n) - 1
    r = np.sqrt(np.maximum(1 - z * z, 0))
    theta = 2 * np.pi * rng.random(n)
    return torch.tensor(np.stack([r * np.cos(theta) * scale, r * np.sin(theta) * scale, z * scale],
                                 axis=1).astype(np.float32))


def test_residual_decoder_shapes_and_range():
    dec = LocalResidualDecoder(z_dim=16, n_local=16)
    B = 6
    e = torch.rand(B, 5)
    p = torch.rand(B, 8)
    p = p / p.sum(-1, keepdim=True)
    center = torch.zeros(B, 3)
    extent = torch.ones(B, 3) * 0.5
    z = torch.randn(B, 16)
    pts = dec(e, p, center, extent, z)
    assert pts.shape == (B, 16, 3)
    assert pts.abs().max() <= 0.25 + 1e-5


def _blob(n=256, seed=0):
    rng = np.random.default_rng(seed)
    axes = rng.uniform(0.4, 1.2, size=3)
    z = 2 * rng.random(n) - 1
    r = np.sqrt(np.maximum(1 - z * z, 0))
    theta = 2 * np.pi * rng.random(n)
    pts = np.stack([r * np.cos(theta) * axes[0], r * np.sin(theta) * axes[1], z * axes[2]], axis=1)
    return torch.tensor(pts.astype(np.float32))


def test_cov_1nna_identical_sets():
    ref = torch.stack([_blob(seed=i) for i in range(24)])
    gen = torch.stack([_blob(seed=100 + i) for i in range(24)])
    res = cov_mmd_1nna(gen, ref)
    assert res["COV"] >= 0.5
    assert 0.25 <= res["1-NNA-CD"] <= 0.75
    assert res["MMD-CD"] < 0.1


def test_cov_detects_shifted_distribution():
    ref = torch.stack([_blob(seed=i) for i in range(16)])
    gen = torch.stack([_blob(seed=i) + 5.0 for i in range(16)])
    res = cov_mmd_1nna(gen, ref)
    assert res["MMD-CD"] > 1.0
    assert res["1-NNA-CD"] > 0.9


def test_mmd_dcd_and_jsd():
    ref = torch.stack([_blob(seed=i) for i in range(8)])
    gen_same = ref.clone()
    gen_far = ref + 5.0
    d_same = mmd_dcd(gen_same, ref)
    d_far = mmd_dcd(gen_far, ref)
    assert d_same < 0.5
    assert d_far > d_same
    j_same = jsd_between_point_cloud_sets(gen_same, ref)
    j_far = jsd_between_point_cloud_sets(gen_far, ref)
    assert j_same < j_far
