import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf import build_octree, normalized_entropy


def _sphere_cloud(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    z = 2 * rng.random(n) - 1
    r = np.sqrt(np.maximum(1 - z * z, 0))
    theta = 2 * np.pi * rng.random(n)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def test_mass_conservation_exact_at_every_depth():
    pc = _sphere_cloud()
    tree = build_octree(pc, max_depth=6)
    for d in range(tree.max_depth + 1):
        cut = tree.cut_at_depth(d)
        assert abs(tree.mass[cut].sum() - 1.0) < 1e-9


def test_child_mass_sums_to_parent():
    pc = _sphere_cloud()
    tree = build_octree(pc, max_depth=5)
    internal = np.nonzero(tree.internal_mask())[0]
    for node in internal:
        children = tree.children[node]
        child_sum = tree.mass[children[children >= 0]].sum()
        assert abs(child_sum - tree.mass[node]) < 1e-9


def test_p_distributions_valid():
    pc = _sphere_cloud(seed=3)
    tree = build_octree(pc, max_depth=6)
    internal = np.nonzero(tree.internal_mask())[0]
    p = tree.p[internal]
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert (p >= 0).all()


def test_normalized_entropy_bounds():
    pc = _sphere_cloud()
    tree = build_octree(pc, max_depth=6)
    internal = np.nonzero(tree.internal_mask())[0]
    h = normalized_entropy(tree.p[internal])
    assert (h >= -1e-9).all() and (h <= 1.0 + 1e-9).all()


def test_single_point_tree():
    pc = np.array([[0.5, 0.5, 0.5]])
    tree = build_octree(pc, max_depth=4)
    assert tree.check_mass_conservation()
    assert tree.n_nodes >= 1


def test_block_embeddings_shape():
    pc = _sphere_cloud(n=500)
    tree = build_octree(pc, max_depth=4)
    e = tree.block_embeddings()
    assert e.shape == (tree.n_nodes, 5)
