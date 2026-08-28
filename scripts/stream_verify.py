"""Milestone 2: streaming verification — throughput, RAM, failure rate, octree stats.

Usage:
    python scripts/stream_verify.py --config chair --n-objects 100
"""

import argparse
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf.data import ShapeNetSDFObjectStream, StreamMode
from eg_hrdf.octree import build_flat_hierarchy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chair")
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=["category", "all"], default="category")
    parser.add_argument("--n-objects", type=int, default=100)
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mode = StreamMode.ALL if args.mode == "all" else StreamMode.CATEGORY
    stream = ShapeNetSDFObjectStream(
        config=args.config,
        split=args.split,
        mode=mode,
        n_points=args.n_points,
        max_objects=args.n_objects,
        seed=args.seed,
    )

    t0 = time.time()
    node_counts, level_counts = [], []
    n_seen = 0
    for record in iter(stream):
        tree = build_flat_hierarchy(record["points"], max_depth=args.depth)
        assert tree.check_mass_conservation(), f"mass conservation failed for {record['model_id']}"
        node_counts.append(tree.node_count())
        level_counts.append([len(tree.levels[d].coords) for d in range(args.depth)])
        n_seen += 1
        if n_seen % 10 == 0:
            elapsed = time.time() - t0
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[{n_seen:>4}/{args.n_objects}] {n_seen / elapsed:.2f} obj/s | "
                  f"peak RSS {rss:.0f} MB | failures {len(stream.failures)}")

    elapsed = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("\n=== Milestone 2 report ===")
    print(f"objects streamed     : {n_seen}")
    print(f"throughput           : {n_seen / elapsed:.2f} obj/s ({elapsed:.1f}s total)")
    print(f"peak RSS             : {rss:.0f} MB")
    print(f"failures             : {len(stream.failures)}")
    for f in stream.failures[:5]:
        print(f"  {f}")
    if node_counts:
        nc = np.array(node_counts)
        lc = np.array(level_counts)
        print(f"nodes/object         : mean {nc.mean():.0f}  median {np.median(nc):.0f}  max {nc.max()}")
        print(f"nodes per depth      : {lc.mean(axis=0).astype(int).tolist()}")
        full = (2 ** args.depth) ** 3
        print(f"leaf occupancy ratio : {lc.mean(axis=0)[-1] / full:.4f} of {full} cells")


if __name__ == "__main__":
    main()
