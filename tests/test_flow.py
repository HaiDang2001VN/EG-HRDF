import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf import FiLMDensityFlowNet, SimplexDensityFlowMatcher


def _random_targets(n=64, seed=0):
    rng = np.random.default_rng(seed)
    p1 = rng.dirichlet(np.ones(8) * 0.3, size=n).astype(np.float32)
    p1[p1 < 1e-3] = 0
    p1 /= p1.sum(axis=1, keepdims=True)
    mass = rng.random(n).astype(np.float32) + 0.05
    e = rng.random((n, 5)).astype(np.float32)
    return torch.tensor(p1), torch.tensor(mass), torch.tensor(e)


def test_network_forward_shapes():
    net = FiLMDensityFlowNet(ctx_dim=16, z_dim=8)
    p_t = torch.rand(10, 8)
    t = torch.rand(10)
    e = torch.rand(10, 5)
    ctx = torch.rand(10, 16)
    z = torch.rand(10, 8)
    logits = net(p_t, t, e, ctx=ctx, z=z)
    assert logits.shape == (10, 8)
    logits_none = net(p_t, t, e, ctx=None, z=None)
    assert logits_none.shape == (10, 8)


def test_loss_finite_and_decreasing():
    torch.manual_seed(0)
    p1, mass, e = _random_targets(n=128)
    net = FiLMDensityFlowNet(z_dim=0)
    matcher = SimplexDensityFlowMatcher(gamma=0.5)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = last = None
    for step in range(300):
        loss = matcher.training_loss(net, p1, e, mass)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first


def test_endpoint_velocity_identity():
    logits = torch.randn(5, 8)
    t = torch.rand(5)
    v = SimplexDensityFlowMatcher.velocity(logits, t)
    p0 = torch.full_like(logits, 1 / 8)
    p1 = v + p0
    assert torch.allclose(p1.sum(-1), torch.ones(5), atol=1e-5)
    assert (p1 >= -1e-5).all()


def test_block_sampling_mixture():
    mass = torch.tensor([0.9, 0.05, 0.05])
    matcher = SimplexDensityFlowMatcher(alpha=1.0, lam=1.0)
    p = matcher.block_sampling_prob(mass)
    assert abs(p.sum().item() - 1.0) < 1e-6
    assert p[0] > p[1]
