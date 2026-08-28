import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf.octree import build_flat_hierarchy, quantize_points
from eg_hrdf.octree.coordinates import decode_coords, encode_coords
from eg_hrdf import build_octree


def _cloud(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    z = 2 * rng.random(n) - 1
    r = np.sqrt(np.maximum(1 - z * z, 0))
    theta = 2 * np.pi * rng.random(n)
    pc = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    pc += 0.02 * rng.standard_normal(pc.shape)
    return pc.astype(np.float32)


def test_flat_mass_conservation_every_depth():
    pc = _cloud()
    tree = build_flat_hierarchy(pc, max_depth=6)
    assert tree.check_mass_conservation()
    for depth, level in tree.levels.items():
        assert abs(level.mass.sum() - 1.0) < 1e-9
        assert np.allclose(level.occupancy.sum(axis=1), 1.0, atol=1e-9)
        assert (level.occupancy >= 0).all()


def test_flat_vs_recursive_oracle():
    pc = _cloud(seed=7)
    unit = (pc + 1.0) / 2.0
    unit = np.clip(unit, 0.0, 1.0 - 1e-7)
    depth = 5
    flat = build_flat_hierarchy(unit, max_depth=depth, lo=0.0, hi=1.0)
    ref = build_octree(unit, max_depth=depth, normalize=False, min_points=0)

    for depth_l in range(depth):
        level = flat.levels.get(depth_l)
        assert level is not None, f"flat missing depth {depth_l}"
        ref_nodes = ref.nodes_at_depth(depth_l)
        ref_cells = {tuple(ref.cell[n]): n for n in ref_nodes}
        flat_cells = {tuple(c): j for j, c in enumerate(level.coords)}
        assert set(ref_cells) == set(flat_cells), f"cell mismatch at depth {depth_l}"
        for cell, n in ref_cells.items():
            j = flat_cells[cell]
            assert abs(level.mass[j] - ref.mass[n]) < 1e-9
            assert np.allclose(level.occupancy[j], ref.p[n], atol=1e-9)


def test_coord_roundtrip():
    rng = np.random.default_rng(0)
    depth = 6
    coords = rng.integers(0, 2 ** depth, size=(100, 3))
    keys = encode_coords(coords, depth)
    back = decode_coords(keys, depth)
    assert np.array_equal(coords, back)


def test_entropy_and_var_bounds():
    pc = _cloud(seed=3)
    tree = build_flat_hierarchy(pc, max_depth=6)
    for level in tree.levels.values():
        assert (level.entropy >= -1e-9).all() and (level.entropy <= 1.0 + 1e-9).all()
        assert (level.occupancy_var >= -1e-12).all()


def test_child_mask_consistency():
    pc = _cloud(seed=5)
    tree = build_flat_hierarchy(pc, max_depth=5)
    for depth in range(tree.max_depth):
        level = tree.levels[depth]
        child_coords = tree.levels[depth + 1].coords if depth + 1 in tree.levels else tree.leaf_coords
        if depth + 1 == tree.max_depth:
            child_coords = tree.leaf_coords
        parents = {tuple(p) for p in (child_coords.astype(np.int64) >> 1)}
        for i, c in enumerate(level.coords):
            has_children = level.child_mask[i].sum() > 0
            assert has_children == (tuple(c) in parents), f"depth {depth} node {tuple(c)}"


def test_quantize_clips():
    pts = np.array([[-1.5, 1.5, 0.0]])
    q = quantize_points(pts, 4)
    assert q.min() >= 0 and q.max() < 16
