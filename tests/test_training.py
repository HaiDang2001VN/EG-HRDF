import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf.network import FiLMDensityFlowNet
from eg_hrdf.octree import build_flat_hierarchy
from eg_hrdf.training import (
    TripleBatcher,
    hierarchical_consistency_loss,
    density_and_hierarchy_loss,
)
from eg_hrdf.training.triple_dataset import TripleBatcher as TB


def _sphere_pc(n=2048, seed=0):
    rng = np.random.default_rng(seed)
    z = 2 * rng.random(n) - 1
    r = np.sqrt(np.maximum(1 - z * z, 0))
    theta = 2 * np.pi * rng.random(n)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(np.float32)


class _FakeReservoir:
    def __init__(self, seed=0):
        pc = _sphere_pc(seed=seed)
        from eg_hrdf.training.triple_dataset import _TreeEntry
        self.entry = _TreeEntry("test", "chair", build_flat_hierarchy(pc, max_depth=6))

    def sample_entry(self):
        return self.entry


def test_hier_loss_zero_for_consistent_predictions():
    torch.manual_seed(0)
    B = 4
    parent_logits = torch.randn(B, 8) * 5
    p_parent = torch.softmax(parent_logits, dim=-1)
    perm = torch.stack([torch.randperm(8) for _ in range(B)])
    child_logits = torch.full((B, 8, 8), -20.0)
    for b in range(B):
        for i in range(8):
            child_logits[b, i, perm[b, i]] = 20.0
    gm = torch.zeros(B, 8, 8)
    for b in range(B):
        for i in range(8):
            gm[b, i, perm[b, i]] = p_parent[b, i] + 1e-3
    loss = hierarchical_consistency_loss(parent_logits, child_logits, gm)
    assert loss.item() >= 0
    assert loss.item() < 0.1


def test_hier_loss_gradients():
    torch.manual_seed(1)
    parent_logits = torch.randn(2, 8, requires_grad=True)
    child_logits = torch.randn(2, 8, 8, requires_grad=True)
    gm = torch.rand(2, 8, 8) + 0.01
    loss = hierarchical_consistency_loss(parent_logits, child_logits, gm)
    loss.backward()
    assert parent_logits.grad is not None
    assert child_logits.grad is not None


def test_combined_loss_finite():
    torch.manual_seed(2)
    out = density_and_hierarchy_loss(
        torch.randn(3, 8), torch.randn(3, 8, 8),
        torch.rand(3, 8), torch.rand(3),
        torch.rand(3, 8, 8), torch.rand(3, 8),
        torch.rand(3, 8, 8),
    )
    total, ld, lh = out
    assert torch.isfinite(total) and torch.isfinite(ld) and torch.isfinite(lh)


def test_triple_batcher_structure():
    batcher = TripleBatcher(_FakeReservoir(seed=1), seed=0)
    t = batcher.sample_triple()
    assert t["parent_p1"].shape == (8,)
    assert abs(t["parent_p1"].sum() - 1.0) < 1e-9
    assert t["child_p1"].shape == (8, 8)
    occupied = t["parent_p1"] > 0
    for o in range(8):
        if occupied[o]:
            assert abs(t["child_p1"][o].sum() - 1.0) < 1e-9
            assert abs(t["grandchild_mass"][o].sum() - t["child_mass"][o]) < 1e-9
        else:
            assert t["child_mass"][o] == 0.0
            assert t["grandchild_mass"][o].sum() == 0.0
    assert t["parent_e"].shape == (5,)
    assert t["child_e"].shape == (8, 5)


def test_triple_batch_shapes():
    batcher = TripleBatcher(_FakeReservoir(seed=2), seed=0)
    batch = batcher.sample_batch(5)
    assert batch["parent_p1"].shape == (5, 8)
    assert batch["child_p1"].shape == (5, 8, 8)
    assert batch["grandchild_mass"].shape == (5, 8, 8)
    assert len(batch["model_ids"]) == 5


def test_streaming_step_backward():
    torch.manual_seed(3)
    net = FiLMDensityFlowNet(z_dim=32)
    z_gen = __import__("eg_hrdf").HierarchicalLatentGenerator(z_dim=32)
    batcher = TripleBatcher(_FakeReservoir(seed=3), seed=0)
    batch = batcher.sample_batch(4)
    B = 4
    z_root = z_gen.root(B, torch.device("cpu"))
    z_c = z_gen.expand(z_root.repeat_interleave(8, dim=0), batch["child_e"].reshape(B * 8, -1))
    p_t_p = 0.5 * torch.full_like(batch["parent_p1"], 0.25) + 0.5 * batch["parent_p1"]
    t_p = torch.full((B,), 0.5)
    p_t_c = 0.5 * torch.full_like(batch["child_p1"].reshape(B * 8, 8), 0.25) + 0.5 * batch["child_p1"].reshape(B * 8, 8)
    t_c = torch.full((B * 8,), 0.5)
    pl = net(p_t_p, t_p, batch["parent_e"], z=z_root)
    cl = net(p_t_c, t_c, batch["child_e"].reshape(B * 8, -1), z=z_c).reshape(B, 8, 8)
    loss, _, _ = density_and_hierarchy_loss(
        pl, cl, batch["parent_p1"], batch["parent_mass"],
        batch["child_p1"], batch["child_mass"], batch["grandchild_mass"])
    loss.backward()
