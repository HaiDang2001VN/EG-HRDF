import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf.evaluation.rendering import render_point_cloud_views


def _sphere(n=512, seed=0):
    rng = np.random.default_rng(seed)
    z = 2 * rng.random(n) - 1
    r = np.sqrt(np.maximum(1 - z * z, 0))
    theta = 2 * np.pi * rng.random(n)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(np.float32)


def test_render_views_shapes_and_range():
    pts = _sphere()
    views = render_point_cloud_views(pts, image_size=64)
    assert views.shape[0] == 16
    assert views.shape[1] == 3
    assert views.shape[-1] == 64
    assert views.min() >= 0.0 and views.max() <= 1.0


def test_render_views_distinct_azimuths():
    pts = _sphere(seed=3)
    views = render_point_cloud_views(pts, image_size=32)
    a = views[0]
    b = views[2]
    assert not torch.allclose(a, b)
