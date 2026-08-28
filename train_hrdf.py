"""EG-HRDF Stage A/B training script.

Examples:
  Synthetic smoke test (no data needed):
    python train_hrdf.py --synthetic --max-shapes 16 --epochs 5 --blocks-per-shape 64

  ShapeNet training:
    python train_hrdf.py --dataroot ./ShapeNetCore.v2.PC15k/ --category chair
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eg_hrdf import (
    AdaptiveDensityScheduler,
    FiLMDensityFlowNet,
    SchedulerConfig,
    SimplexDensityFlowMatcher,
    build_octree,
    chamfer_distance,
)
from eg_hrdf.data import OctreeNodeDataset
from eg_hrdf.hier_latent import HierarchicalLatentGenerator


def synthetic_point_clouds(n_shapes: int, n_points: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    clouds = []
    for i in range(n_shapes):
        kind = i % 3
        u = rng.random(n_points)
        theta = 2 * np.pi * rng.random(n_points)
        z = 2 * rng.random(n_points) - 1
        r = np.sqrt(np.maximum(1 - z * z, 0))
        if kind == 0:
            pts = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
        elif kind == 1:
            rr = 0.6 + 0.2 * rng.random(n_points)
            pts = np.stack([rr * np.cos(theta), rr * np.sin(theta), 0.3 * z], axis=1)
        else:
            phi = 2 * np.pi * rng.random(n_points)
            pts = np.stack(
                [0.4 * np.cos(theta), 0.4 * np.sin(theta), 0.15 * np.sin(phi) + 0.3 * np.cos(theta * 3)],
                axis=1,
            )
            pts += 0.02 * rng.standard_normal(pts.shape)
        clouds.append(pts.astype(np.float32))
    return clouds


def load_shapenet_clouds(dataroot: str, category: str, max_shapes: int, npoints: int) -> list:
    from datasets.shapenet_data_pc import ShapeNet15kPointClouds

    ds = ShapeNet15kPointClouds(
        root_dir=dataroot,
        categories=[category],
        split="train",
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.0,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True,
    )
    n = len(ds) if max_shapes <= 0 else min(max_shapes, len(ds))
    return [ds[i]["train_points"].astype(np.float32) for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="")
    parser.add_argument("--category", default="chair")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--blocks-per-shape", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--z-dim", type=int, default=0)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="output/train_hrdf")
    parser.add_argument("--eval-every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.synthetic:
        clouds = synthetic_point_clouds(max(args.max_shapes, 8), args.npoints, seed=args.seed)
    else:
        assert args.dataroot, "--dataroot required unless --synthetic"
        clouds = load_shapenet_clouds(args.dataroot, args.category, args.max_shapes, args.npoints)

    print(f"Building octrees (depth={args.depth}) for {len(clouds)} shapes...")
    octrees = [build_octree(pc, max_depth=args.depth) for pc in clouds]
    for i, tree in enumerate(octrees):
        assert tree.check_mass_conservation(), f"mass conservation violated for shape {i}"
    print("Mass conservation verified for all shapes.")

    dataset = OctreeNodeDataset(
        octrees,
        blocks_per_shape=args.blocks_per_shape,
        alpha=args.alpha,
        lam=args.lam,
    )
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=OctreeNodeDataset.collate_fn)

    net = FiLMDensityFlowNet(ctx_dim=0, z_dim=args.z_dim).to(device)
    matcher = SimplexDensityFlowMatcher(gamma=args.gamma, alpha=args.alpha, lam=args.lam)
    z_gen = HierarchicalLatentGenerator(z_dim=args.z_dim) if args.z_dim > 0 else None
    if z_gen is not None:
        z_gen.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        net.train()
        for batch in loader:
            p1 = batch["p1"].to(device)
            mass = batch["mass"].to(device)
            e = batch["e"].to(device)
            z = None
            if z_gen is not None:
                shape_idx = batch["shape_idx"].to(device)
                root_z = z_gen.root(int(shape_idx.max()) + 1, device)
                z = root_z[shape_idx]
            loss = matcher.training_loss(net, p1, e, mass, ctx=None, z=z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % 50 == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            net.eval()
            scheduler = AdaptiveDensityScheduler(
                net,
                SchedulerConfig(max_depth=args.depth, n_points=args.npoints, budget=None, score_mode="entropy"),
            )
            points, stats = scheduler.generate(device)
            if args.synthetic and points.shape[0] > 0:
                ref = synthetic_point_clouds(1, args.npoints, seed=args.seed + 1000)[0]
                gen_t = torch.tensor(points)
                ref_t = torch.tensor((ref - ref.min(0)) / max(ref.max() - ref.min(), 1e-9))
                d1, d2 = chamfer_distance(gen_t, ref_t)
                print(f"[eval] epoch {epoch} K={stats.evaluated} leaves={stats.leaves} "
                      f"macro={stats.macro_voxels} pruned={stats.pruned} "
                      f"chamfer={d1.mean() + d2.mean():.4f}")
            else:
                print(f"[eval] epoch {epoch} K={stats.evaluated} leaves={stats.leaves} "
                      f"macro={stats.macro_voxels} pruned={stats.pruned}")
            torch.save({"net": net.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "hrdf_latest.pth"))

    print("done")


if __name__ == "__main__":
    main()
