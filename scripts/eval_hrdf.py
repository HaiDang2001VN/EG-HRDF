"""Generation + evaluation harness: adaptive-budget inference and generative metrics.

Usage:
    python scripts/eval_hrdf.py --ckpt output/m4_deterministic/hrdf_stream_latest.pth \
        --config chair --split val --n-gen 64 --n-ref 128 \
        --budgets 1.0 0.5 0.25 0.1
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf import AdaptiveDensityScheduler, FiLMDensityFlowNet, PerceiverDensityFlowNet, SchedulerConfig
from eg_hrdf.data import ShapeNetSDFObjectStream, StreamMode
from eg_hrdf.evaluation import cov_mmd_1nna, jsd_between_point_cloud_sets, mmd_dcd
from eg_hrdf.hier_latent import HierarchicalLatentGenerator


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    z_dim = args["z_dim"] if args["z_mode"] != "none" else 0
    text_dim = 512 if args.get("text_embeddings") else 0
    if args.get("arch") == "perceiver":
        net = PerceiverDensityFlowNet(branch=args.get("branch", 2), n_blocks=args.get("n_blocks", 2),
                                      dim=args.get("dim", 256), z_dim=z_dim, text_dim=text_dim)
    else:
        net = FiLMDensityFlowNet(z_dim=z_dim, text_dim=text_dim)
    net.load_state_dict(ckpt["net"])
    net.to(device).eval()
    z_gen = None
    if args["z_mode"] == "hier":
        z_gen = HierarchicalLatentGenerator(z_dim=args["z_dim"])
        z_gen.load_state_dict(ckpt["z_gen"])
        z_gen.to(device).eval()
    hash_ctx = None
    if args.get("use_hash"):
        from eg_hrdf import SpatialHashContext
        hash_ctx = SpatialHashContext(n_levels=2, out_dim=32,
                                      n_neighbors=args.get("hash_neighbors", 6),
                                      branch=args.get("branch", 2))
        hash_ctx.load_state_dict(ckpt["hash_ctx"])
        hash_ctx.to(device).eval()
    return net, z_gen, hash_ctx, args


def k_full_for_category(depth: int, mean_leaves: float) -> int:
    return int(mean_leaves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="chair")
    parser.add_argument("--split", default="val")
    parser.add_argument("--mode", choices=["category", "all"], default="category")
    parser.add_argument("--n-gen", type=int, default=64)
    parser.add_argument("--n-ref", type=int, default=128)
    parser.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    parser.add_argument("--k-full", type=int, default=0,
                        help="full-resolution node count; 0 = measured on the fly (1.0 budget run)")
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="output/eval_hrdf.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, z_gen, hash_ctx, train_args = load_model(args.ckpt, device)
    ctx_fn = None
    if hash_ctx is not None:
        def ctx_fn(cells, depth_t):
            return hash_ctx(torch.as_tensor(np.asarray(cells), dtype=torch.long, device=device),
                            torch.as_tensor(depth_t, dtype=torch.long, device=device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"collecting {args.n_ref} reference clouds from {args.config} {args.split}...")
    ref_stream = ShapeNetSDFObjectStream(
        config=args.config, split=args.split,
        mode=StreamMode.ALL if args.mode == "all" else StreamMode.CATEGORY,
        n_points=args.n_points, max_objects=args.n_ref, seed=args.seed + 1,
    )
    refs = []
    for rec in iter(ref_stream):
        refs.append(torch.tensor(rec["points"]))
    ref_set = torch.stack(refs)
    print(f"reference set: {tuple(ref_set.shape)}")

    results = []
    k_full = args.k_full
    for rho in args.budgets:
        gen_set = []
        t0 = time.time()
        total_evaluated = 0
        for g in range(args.n_gen):
            cfg = SchedulerConfig(
                max_depth=train_args.get("depth", args.depth), branch=train_args.get("branch", 2),
                n_points=args.n_points, score_mode="entropy",
                domain=(-1.0, 1.0),
            )
            sched = AdaptiveDensityScheduler(net, cfg)
            if rho < 1.0:
                if k_full == 0:
                    probe_cfg = SchedulerConfig(max_depth=cfg.max_depth, branch=cfg.branch,
                                                n_points=args.n_points, domain=(-1.0, 1.0))
                    _, probe_stats = AdaptiveDensityScheduler(net, probe_cfg).generate(device)
                    k_full = max(probe_stats.evaluated, 1)
                cfg.budget = max(1, int(round(rho * k_full)))
            z0 = z_gen.root(1, device) if z_gen is not None else None
            z_fn = (lambda zp, e: z_gen.expand(zp, e)) if z_gen is not None else None
            pts, stats = sched.generate(
                device, z_root=z0, z_fn=z_fn, ctx_fn=ctx_fn,
            )
            total_evaluated += stats.evaluated
            gen_set.append(torch.tensor(pts))
        runtime = time.time() - t0
        gen_set = torch.stack(gen_set)

        metrics = cov_mmd_1nna(gen_set, ref_set)
        metrics["MMD-DCD"] = mmd_dcd(gen_set, ref_set)
        metrics["JSD"] = jsd_between_point_cloud_sets(gen_set, ref_set)
        row = {
            "rho": rho,
            "budget_K": cfg.budget if rho < 1.0 else k_full,
            "mean_evaluated": total_evaluated / args.n_gen,
            "runtime_s": runtime,
            **metrics,
        }
        results.append(row)
        print(json.dumps(row, indent=None))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
