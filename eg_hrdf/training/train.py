"""Streaming EG-HRDF trainer (data.md 26-28): node/triple-level batches from ShapeNetSDF.

Usage:
    python -m eg_hrdf.training.train --config chair --steps 2000 --branch 4 --z-mode hier
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eg_hrdf.data import ShapeNetSDFObjectStream, StreamMode
from eg_hrdf.hier_latent import HierarchicalLatentGenerator
from eg_hrdf.hash_context import SpatialHashContext
from eg_hrdf.network import FiLMDensityFlowNet, PerceiverDensityFlowNet
from eg_hrdf.training import TripleBatcher, TripleReservoir, density_and_hierarchy_loss


def build_net(args):
    ctx_dim = 32 if args.use_hash else 0
    if args.arch == "perceiver":
        return PerceiverDensityFlowNet(
            branch=args.branch, n_blocks=args.n_blocks, dim=args.dim,
            z_dim=args.z_dim if args.z_mode != "none" else 0,
            text_dim=512 if args.text_embeddings else 0,
            ctx_dim=ctx_dim,
        )
    return FiLMDensityFlowNet(z_dim=args.z_dim if args.z_mode != "none" else 0,
                              text_dim=512 if args.text_embeddings else 0,
                              ctx_dim=ctx_dim)


def build_z(args, B, device, z_gen):
    if args.z_mode == "none":
        return None, None
    if args.z_mode == "independent":
        return torch.randn(B, args.z_dim, device=device), None
    return z_gen.root(B, device), None


def interpolate_simplex(p1: torch.Tensor, device: torch.device, flow_mode: str = "simplex") -> tuple:
    B = p1.shape[0]
    if flow_mode == "direct":
        return p1, torch.ones(B, device=device, dtype=torch.float32)
    t = torch.rand(B, device=device, dtype=torch.float32)[:, None]
    p0 = torch.full_like(p1, 1.0 / p1.shape[-1])
    p_t = (1.0 - t) * p0 + t * p1
    return p_t, t.squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chair")
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=["category", "all"], default="category")
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--arch", choices=["perceiver", "film"], default="perceiver")
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--reservoir", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-triples", type=int, default=4)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lambda-hier", type=float, default=0.1)
    parser.add_argument("--flow-mode", choices=["simplex", "direct"], default="simplex")
    parser.add_argument("--use-hash", action="store_true")
    parser.add_argument("--hash-neighbors", type=int, default=6, choices=[6, 18, 26])
    parser.add_argument("--z-mode", choices=["none", "independent", "hier"], default="hier")
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--text-embeddings", default="", help="dir with caption_embeddings.npy + caption_ids.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="output/train_hrdf_stream")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          (args.device if args.device != "auto" else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    n_children = args.branch ** 3

    text_embs = None
    text_ids = None
    if args.text_embeddings:
        text_embs = np.load(os.path.join(args.text_embeddings, "caption_embeddings.npy"))
        import json
        text_ids = json.load(open(os.path.join(args.text_embeddings, "caption_ids.json")))

    stream = ShapeNetSDFObjectStream(
        config=args.config,
        split=args.split,
        mode=StreamMode.ALL if args.mode == "all" else StreamMode.CATEGORY,
        n_points=args.n_points,
        seed=args.seed,
    )
    reservoir = TripleReservoir(stream, size=args.reservoir, depth=args.depth,
                                branch=args.branch, seed=args.seed)
    batcher = TripleBatcher(reservoir, seed=args.seed)
    reservoir.start()

    net = build_net(args).to(device)
    z_gen = HierarchicalLatentGenerator(z_dim=args.z_dim) if args.z_mode == "hier" else None
    if z_gen is not None:
        z_gen.to(device)
    hash_ctx = None
    if args.use_hash:
        hash_ctx = SpatialHashContext(n_levels=2, out_dim=32, n_neighbors=args.hash_neighbors,
                                      branch=args.branch).to(device)
    opt = torch.optim.Adam(list(net.parameters()) + (list(z_gen.parameters()) if z_gen else [])
                           + (list(hash_ctx.parameters()) if hash_ctx else []), lr=args.lr)

    t0 = time.time()
    running = []
    while reservoir.seen < 3:
        time.sleep(0.5)
    print(f"reservoir ready ({reservoir.seen} objects streamed); "
          f"training {args.arch} branch={args.branch} (n_children={n_children}) depth={args.depth} on {device}")

    for step in range(1, args.steps + 1):
        net.train()
        opt.zero_grad()
        step_loss = 0.0
        for _ in range(args.accum):
            batch = batcher.sample_batch(args.batch_triples)
            text = None
            if text_embs is not None:
                idx = torch.tensor(
                    [text_ids.get(m, -1) for m in batch["model_ids"]], dtype=torch.long
                )
                text = torch.zeros(len(batch["model_ids"]), text_embs.shape[1], dtype=torch.float32)
                known = idx >= 0
                if known.any():
                    text[known] = torch.tensor(text_embs[idx[known]], dtype=torch.float32)
                text = text.to(device)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            B = batch["parent_mass"].shape[0]

            z_p, _ = build_z(args, B, device, z_gen)
            z_c = None
            if args.z_mode == "independent":
                z_c = torch.randn(B * n_children, args.z_dim, device=device)
            elif args.z_mode == "hier":
                z_c = z_gen.expand(z_p.repeat_interleave(n_children, dim=0),
                                   batch["child_e"].reshape(B * n_children, -1))

            p_t_p, t_p = interpolate_simplex(batch["parent_p1"], device, args.flow_mode)
            p_t_c, t_c = interpolate_simplex(batch["child_p1"].reshape(B * n_children, n_children), device, args.flow_mode)
            ctx_p = None
            ctx_c = None
            if hash_ctx is not None:
                ctx_p = hash_ctx(batch["parent_cell"], batch["parent_depth"])
                cc = batch["child_cell"].reshape(B * n_children, 3)
                cd = batch["child_depth"].repeat_interleave(n_children, dim=0)
                ctx_c = hash_ctx(cc, cd)
            parent_logits = net(p_t_p, t_p, batch["parent_e"], ctx=ctx_p, z=z_p, text=text)
            children_logits = net(p_t_c, t_c, batch["child_e"].reshape(B * n_children, -1),
                                  ctx=ctx_c,
                                  z=z_c,
                                  text=text.repeat_interleave(n_children, dim=0) if text is not None else None,
                                  ).reshape(B, n_children, n_children)

            loss, l_d, l_h = density_and_hierarchy_loss(
                parent_logits, children_logits,
                batch["parent_p1"], batch["parent_mass"],
                batch["child_p1"], batch["child_mass"],
                batch["grandchild_mass"],
                gamma=args.gamma, lambda_hier=args.lambda_hier,
            )
            (loss / args.accum).backward()
            step_loss += loss.item() / args.accum
        opt.step()

        running.append(step_loss)
        if step % args.log_every == 0:
            rss = 0
            try:
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except Exception:
                pass
            print(f"step {step:>6} loss {np.mean(running[-args.log_every:]):.4f} "
                  f"({(time.time() - t0) / step:.2f}s/step, obj streamed {reservoir.seen}, "
                  f"RSS {rss:.0f} MB)")
            torch.save({"net": net.state_dict(), "z_gen": z_gen.state_dict() if z_gen else None,
                        "hash_ctx": hash_ctx.state_dict() if hash_ctx else None,
                        "args": vars(args)},
                       os.path.join(args.out_dir, "hrdf_stream_latest.pth"))

    reservoir.stop()
    print("done")


if __name__ == "__main__":
    main()
