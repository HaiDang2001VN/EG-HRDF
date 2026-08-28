"""Multi-view point-cloud rendering for semantic evaluation (data.md 35)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

ELEVATIONS = (20.0, -20.0)
AZIMUTHS = tuple(np.linspace(0, 360, 8, endpoint=False))


def render_point_cloud_views(
    points: np.ndarray,
    azimuths: tuple = AZIMUTHS,
    elevations: tuple = ELEVATIONS,
    image_size: int = 224,
    point_size: float = 0.5,
    elev_scale: float = 2.5,
) -> torch.Tensor:
    """Renders a point cloud from fixed views.

    points: (N, 3) in [-1, 1]. Returns (V, 3, H, W) float in [0, 1].
    """
    import matplotlib
    matplotlib.use("Agg")

    views = []
    pts = np.asarray(points, dtype=np.float32)
    for elev in elevations:
        for az in azimuths:
            fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=elev, azim=az)
            r = elev_scale
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(-r, r)
            ax.set_axis_off()
            ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=point_size, c=pts[:, 2], cmap="viridis")
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.float32) / 255.0
            plt.close(fig)
            views.append(torch.tensor(buf).permute(2, 0, 1))
    return torch.stack(views)
