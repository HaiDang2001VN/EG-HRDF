"""Streaming EG-HRDF trainer (data.md 26-28): node/triple-level batches from ShapeNetSDF.

Usage:
    python -m eg_hrdf.training.train --config chair --steps 2000 --z-mode hier
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
from eg_hrdf.network import FiLMDensityFlowNet
from eg_hrdf.training import TripleBatcher, TripleReservoir, density_and_hierarchy_loss


def build_z(args, batch, device, z_gen):
    B = batch["parent_mass"].shape[0]
    if args.z_mode == "none":
        return None, None, None
    if args.z_mode == "independent":
        z_p = torch.randn(B, args.z_dim, device=device)
        z_c = torch.randn(B * 8, args.z_dim, device=device)
        return z_p, z_c, None
    root = z_gen.root(B, device)
    child_e = batch["child_e"].reshape(B * 8, -1).to(device)
    z_p = root
    z_c = z_gen.expand(root.repeat_interleave(8, dim=0), child_e)
    return z_p, z_c, root


def interpolate_simplex(p1: torch.Tensor, device: torch.Tensor) -> tuple:
    B = p1.shape[0]
    t = torch.rand(B, device=device, dtype=torch.float32)[:, None]
    p0 = torch.full_like(p1, 1.0 / 8.0)
    p_t = (1.0 - t) * p0 + t * p1
    return p_t, t.squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chair")
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=["category", "all"], default="category")
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--reservoir", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-triples", type=int, default=8)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lambda-hier", type=float, default=0.1)
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
    reservoir = TripleReservoir(stream, size=args.reservoir, depth=args.depth, seed=args.seed)
    batcher = TripleBatcher(reservoir, seed=args.seed)
    reservoir.start()

    net = FiLMDensityFlowNet(z_dim=args.z_dim if args.z_mode != "none" else 0,
                             text_dim=512 if text_embs is not None else 0).to(device)
    z_gen = HierarchicalLatentGenerator(z_dim=args.z_dim) if args.z_mode == "hier" else None
    if z_gen is not None:
        z_gen.to(device)
    opt = torch.optim.Adam(list(net.parameters()) + (list(z_gen.parameters()) if z_gen else []), lr=args.lr)

    t0 = time.time()
    running = []
    while reservoir.seen < 3:
        time.sleep(0.5)
    print(f"reservoir ready ({reservoir.seen} objects streamed); training on {device}")

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

            z_p, z_c, _ = build_z(args, batch, device, z_gen)
            z_all = None
            if z_p is not None:
                z_all = torch.cat([z_p, z_c], dim=0)

            p_t_p, t_p = interpolate_simplex(batch["parent_p1"], device)
            p_t_c, t_c = interpolate_simplex(batch["child_p1"].reshape(B * 8, 8), device)
            parent_logits = net(p_t_p, t_p, batch["parent_e"], z=z_p, text=text)
            children_logits = net(p_t_c, t_c, batch["child_e"].reshape(B * 8, -1),
                                  z=z_c, text=text.repeat_interleave(8, dim=0) if text is not None else None).reshape(B, 8, 8)

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
                        "args": vars(args)},
                       os.path.join(args.out_dir, "hrdf_stream_latest.pth"))

    reservoir.stop()
    print("done")


if __name__ == "__main__":
    main()
