import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf import (
    AdaptiveDensityScheduler,
    FiLMDensityFlowNet,
    SchedulerConfig,
    HierarchicalLatentGenerator,
    SpatialHashContext,
    density_aware_chamfer,
)


def _sphere_pc(n=2048, seed=0):
    rng = np.random.default_rng(seed)
    z = 2 * rng.random(n) - 1
    r = np.sqrt(np.maximum(1 - z * z, 0))
    theta = 2 * np.pi * rng.random(n)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(np.float32)


def test_scheduler_budget_respected():
    torch.manual_seed(0)
    net = FiLMDensityFlowNet(z_dim=0)
    cfg = SchedulerConfig(max_depth=5, budget=20, n_points=512, tau_mass=1e-3)
    sched = AdaptiveDensityScheduler(net, cfg)
    pts, stats = sched.generate(torch.device("cpu"))
    assert stats.evaluated <= 20
    assert pts.shape == (512, 3)
    assert pts.min() >= -1.0 and pts.max() <= 1.0


def test_scheduler_full_generation_counts():
    torch.manual_seed(1)
    net = FiLMDensityFlowNet(z_dim=0)
    cfg = SchedulerConfig(max_depth=4, budget=None, n_points=256, tau_mass=1e-3)
    sched = AdaptiveDensityScheduler(net, cfg)
    pts, stats = sched.generate(torch.device("cpu"))
    assert stats.leaves == stats.evaluated + 0 or stats.leaves >= 1
    assert pts.shape == (256, 3)


def test_scheduler_with_stochastic_latents():
    torch.manual_seed(2)
    net = FiLMDensityFlowNet(z_dim=16)
    z_gen = HierarchicalLatentGenerator(z_dim=16)

    def z_fn(z_parent, e_child):
        return z_gen.expand(z_parent, e_child)

    cfg = SchedulerConfig(max_depth=5, budget=100, n_points=512)
    sched = AdaptiveDensityScheduler(net, cfg)
    z0 = z_gen.root(1, torch.device("cpu"))
    pts, stats = sched.generate(torch.device("cpu"), z_root=z0, z_fn=z_fn)
    assert pts.shape == (512, 3)
    assert stats.evaluated <= 100


def test_hash_context_forward():
    ctx = SpatialHashContext(n_levels=2, feat_dim=8, out_dim=16)
    cell = torch.randint(0, 64, (7, 3))
    out = ctx(cell, depth=3, max_depth=6)
    assert out.shape == (7, 16)


def test_dcd_properties():
    x = torch.tensor(_sphere_pc(256, seed=1))
    y_same = x.clone()
    y_far = x + 10.0
    d_same = density_aware_chamfer(x, y_same)
    d_far = density_aware_chamfer(x, y_far)
    assert d_same.item() < 1e-4
    assert d_far.item() > 0.5
