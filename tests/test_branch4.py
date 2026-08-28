import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf import (
    AdaptiveDensityScheduler,
    PerceiverDensityFlowNet,
    SchedulerConfig,
    HierarchicalLatentGenerator,
)
from eg_hrdf.network import FiLMDensityFlowNet
from eg_hrdf.octree import build_flat_hierarchy, quantize_points
from eg_hrdf.training.triple_dataset import TripleBatcher, _TreeEntry

B = 4


def _cloud(n=4096, seed=0):
    rng = np.random.default_rng(seed)
    z = 2 * rng.random(n) - 1
    r = np.sqrt(np.maximum(1 - z * z, 0))
    theta = 2 * np.pi * rng.random(n)
    pc = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    return (pc + 0.02 * rng.standard_normal(pc.shape)).astype(np.float32)


def test_branch4_mass_conservation():
    pc = _cloud()
    tree = build_flat_hierarchy(pc, max_depth=3, branch=4)
    assert tree.check_mass_conservation()
    for depth, level in tree.levels.items():
        assert abs(level.mass.sum() - 1.0) < 1e-9
        assert np.allclose(level.occupancy.sum(axis=1), 1.0, atol=1e-9)
        assert level.occupancy.shape[1] == 64


def test_branch4_occupancy_matches_direct_counting():
    pc = _cloud(seed=3)
    depth, branch = 2, B
    tree = build_flat_hierarchy(pc, max_depth=depth, branch=branch)
    q = quantize_points(pc, depth, branch)
    from collections import Counter
    direct = Counter(map(tuple, q))
    level = tree.levels[depth - 1]
    for i, cell in enumerate(level.coords):
        n_direct = sum(v for k, v in direct.items() if tuple(np.array(k) // branch) == tuple(cell))
        assert abs(level.counts[i] - n_direct) == 0
        child_counts = level.occupancy[i] * level.counts[i]
        for o in range(branch ** 3):
            occ_cell = branch * cell + np.array([o % branch, (o // branch) % branch, (o // branch ** 2) % branch])
            n_child = direct.get(tuple(occ_cell), 0)
            assert abs(child_counts[o] - n_child) < 1e-6


def test_branch4_child_mass_sums():
    branch = B
    pc = _cloud(seed=5)
    tree = build_flat_hierarchy(pc, max_depth=3, branch=4)
    for depth in range(tree.max_depth - 1):
        level = tree.levels[depth]
        child_level = tree.levels[depth + 1]
        child_parent = child_level.coords // branch
        parent_of = {tuple(p): i for i, p in enumerate(level.coords)}
        sums = np.zeros(len(level.coords))
        for j, p in enumerate(child_parent):
            sums[parent_of[tuple(p)]] += child_level.mass[j]
        assert np.allclose(sums, level.mass, atol=1e-9)


def test_perceiver_forward_shapes():
    for branch in (2, 4):
        net = PerceiverDensityFlowNet(branch=branch, z_dim=16, text_dim=512)
        n = branch ** 3
        out = net(torch.rand(7, n), torch.rand(7), torch.rand(7, 5),
                  z=torch.randn(7, 16), text=torch.randn(7, 512))
        assert out.shape == (7, n)
        probs = torch.softmax(out, dim=-1)
        assert torch.allclose(probs.sum(-1), torch.ones(7), atol=1e-5)


def test_perceiver_film_equiv_signature():
    film = FiLMDensityFlowNet(z_dim=8)
    out = film(torch.rand(3, 8), torch.rand(3), torch.rand(3, 5), z=torch.randn(3, 8))
    assert out.shape == (3, 8)


def test_branch4_perceiver_training_step():
    torch.manual_seed(0)
    net = PerceiverDensityFlowNet(branch=4, z_dim=32)
    z_gen = HierarchicalLatentGenerator(z_dim=32)
    pc = _cloud(seed=7)
    reservoir = _FakeReservoir4()
    batcher = TripleBatcher(reservoir, seed=0)
    batch = batcher.sample_batch(2)
    n = 64
    B = 2
    z_root = z_gen.root(B, torch.device("cpu"))
    z_c = z_gen.expand(z_root.repeat_interleave(n, dim=0), batch["child_e"].reshape(B * n, -1))
    p_t_p = 0.5 * torch.full_like(batch["parent_p1"], 1.0 / n) + 0.5 * batch["parent_p1"]
    t_p = torch.full((B,), 0.5)
    p_t_c = 0.5 * torch.full_like(batch["child_p1"].reshape(B * n, n), 1.0 / n) + 0.5 * batch["child_p1"].reshape(B * n, n)
    t_c = torch.full((B * n,), 0.5)
    pl = net(p_t_p, t_p, batch["parent_e"], z=z_root)
    cl = net(p_t_c, t_c, batch["child_e"].reshape(B * n, -1), z=z_c).reshape(B, n, n)
    from eg_hrdf.training import density_and_hierarchy_loss
    loss, _, _ = density_and_hierarchy_loss(
        pl, cl, batch["parent_p1"], batch["parent_mass"],
        batch["child_p1"], batch["child_mass"], batch["grandchild_mass"])
    loss.backward()


class _FakeReservoir4:
    branch = 4

    def __init__(self):
        pc = _cloud(seed=11)
        tree = build_flat_hierarchy(pc, max_depth=3, branch=4)
        self.entry = _TreeEntry("t", "chair", tree)

    def sample_entry(self):
        return self.entry


def test_branch4_triple_batcher():
    batcher = TripleBatcher(_FakeReservoir4(), seed=0)
    t = batcher.sample_triple()
    assert t["parent_p1"].shape == (64,)
    assert abs(t["parent_p1"].sum() - 1.0) < 1e-9
    assert t["child_p1"].shape == (64, 64)
    assert t["grandchild_mass"].shape == (64, 64)
    occupied = t["parent_p1"] > 0
    for o in range(64):
        if occupied[o]:
            assert abs(t["grandchild_mass"][o].sum() - t["child_mass"][o]) < 1e-9


def test_branch4_scheduler_budget_and_domain():
    torch.manual_seed(0)
    net = PerceiverDensityFlowNet(branch=4)
    cfg = SchedulerConfig(max_depth=3, branch=4, budget=15, n_points=512, tau_mass=1e-3)
    pts, stats = AdaptiveDensityScheduler(net, cfg).generate(torch.device("cpu"))
    assert stats.evaluated <= 15
    assert pts.shape == (512, 3)
    assert pts.min() >= -1.0 and pts.max() <= 1.0
