"""WS4 / data.md 41: entropy-validity study.

Labels every octree node as surface-crossing or not using SDF values of uniformly
sampled points, then evaluates normalized entropy h_B as a boundary predictor:
P(y=1|h) histogram, ROC-AUC (entropy vs occupancy variance), calibration.

Usage:
    python scripts/entropy_validity.py --config chair --n-objects 60 --depth 6 --branch 4
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf.data import ShapeNetSDFObjectStream, StreamMode
from eg_hrdf.octree import build_flat_hierarchy
from eg_hrdf.octree.coordinates import encode_coords, quantize_points


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chair")
    parser.add_argument("--n-objects", type=int, default=60)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="output/entropy_validity.json")
    args = parser.parse_args()

    stream = ShapeNetSDFObjectStream(
        config=args.config, split="train", mode=StreamMode.CATEGORY,
        n_points=args.n_points, with_sdf=True, max_objects=args.n_objects, seed=args.seed,
    )

    b = args.branch
    n_child = b ** 3
    h_all, v_all, y_all = [], [], []
    per_depth_pos = {}

    n_seen = 0
    for rec in iter(stream):
        sdf_pts = np.concatenate([rec["sdf_points"], rec["points"]], axis=0)
        sdf = np.concatenate([rec["sdf"], np.zeros(len(rec["points"]))])
        sdf_pts = sdf_pts.astype(np.float64)

        tree = build_flat_hierarchy(rec["points"], max_depth=args.depth, branch=b)

        for depth in range(1, args.depth):
            level = tree.levels.get(depth)
            if level is None:
                continue
            q = quantize_points(sdf_pts, depth, b)
            keys = encode_coords(q, depth, b)
            order = np.argsort(keys)
            sorted_keys = keys[order]
            sorted_sdf = sdf[order]
            uniq, start = np.unique(sorted_keys, return_index=True)
            mn = np.minimum.reduceat(sorted_sdf, start)
            mx = np.maximum.reduceat(sorted_sdf, start)
            crosses = (mn <= 0) & (mx >= 0)
            cell_cross = dict(zip(uniq.tolist(), crosses.tolist()))

            for i, cell in enumerate(level.coords):
                key = int(encode_coords(cell[None], depth, b)[0])
                if key not in cell_cross:
                    continue
                h_all.append(float(level.entropy[i]))
                v_all.append(float(level.occupancy_var[i]))
                y_all.append(1 if cell_cross[key] else 0)
                per_depth_pos.setdefault(depth, [0, 0])
                per_depth_pos[depth][int(cell_cross[key])] += 1
        n_seen += 1
        if n_seen % 10 == 0:
            print(f"processed {n_seen}/{args.n_objects} objects", flush=True)

    h_all = np.array(h_all)
    v_all = np.array(v_all)
    y_all = np.array(y_all)

    bins = np.linspace(0, 1, 11)
    digitized = np.digitize(h_all, bins) - 1
    hist = []
    for k in range(10):
        mask = digitized == k
        n_bin = int(mask.sum())
        p = float(y_all[mask].mean()) if n_bin else None
        hist.append({"h_bin": [bins[k], bins[k + 1]], "n": n_bin, "P(surface|h)": p})

    auc_h = roc_auc(h_all, y_all)
    auc_v = roc_auc(v_all, y_all)

    result = {
        "config": vars(args),
        "n_objects": n_seen,
        "n_nodes": len(y_all),
        "positive_rate": float(y_all.mean()),
        "auc_entropy": auc_h,
        "auc_occupancy_var": auc_v,
        "calibration_by_h_bin": hist,
        "positives_per_depth": {str(k): v for k, v in per_depth_pos.items()},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("n_nodes", "positive_rate", "auc_entropy", "auc_occupancy_var")}, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
