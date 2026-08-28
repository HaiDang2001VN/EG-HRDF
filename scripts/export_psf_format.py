"""Bounded export: ShapeNetSDF stream -> PSF-format point clouds (data.md WS5).

Writes <out>/<category>/<split>/<model_id>.npy with (16384, 3) float32 points.
This is the single deliberate materialization exception (equal-data PSF baseline).

Usage:
    python scripts/export_psf_format.py --config chair --splits train val --out data/psf_export
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf.data import ShapeNetSDFObjectStream, StreamMode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chair")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--mode", choices=["category", "all"], default="category")
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--out", default="data/psf_export")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for split in args.splits:
        stream = ShapeNetSDFObjectStream(
            config=args.config, split=split,
            mode=StreamMode.ALL if args.mode == "all" else StreamMode.CATEGORY,
            n_points=args.n_points, max_objects=args.max_objects, seed=args.seed,
        )
        out_dir = os.path.join(args.out, args.config, split)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for rec in iter(stream):
            np.save(os.path.join(out_dir, rec["model_id"] + ".npy"), rec["points"].astype(np.float32))
            n += 1
            if n % 200 == 0:
                print(f"[{split}] exported {n}", flush=True)
        print(f"[{split}] done: {n} objects -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
