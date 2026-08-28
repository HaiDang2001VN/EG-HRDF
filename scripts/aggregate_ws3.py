"""Aggregate WS3 eval JSONs into the quality-vs-compute table and figure (data.md 40)."""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VARIANTS = {
    "none": "EG-HRDF (no z)",
    "independent": "EG-HRDF + indep. $z_B$",
    "hier": "EG-HRDF + hier. $z_B$",
}

METRICS = ["MMD-CD", "COV", "1-NNA-CD", "JSD", "MMD-DCD"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="output")
    parser.add_argument("--out-dir", default="output/ws3_report")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = {}
    for key, label in VARIANTS.items():
        path = os.path.join(args.eval_dir, f"eval_ws3_{key}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data[key] = json.load(f)["results"]

    table_lines = []
    header = f"{'variant':<22}{'rho':>6}{'K':>7}" + "".join(f"{m:>10}" for m in METRICS)
    table_lines.append(header)
    for key, rows in data.items():
        for row in rows:
            k = row["budget_K"] if row["budget_K"] else "full"
            vals = "".join(f"{row[m]:>10.4f}" for m in METRICS)
            table_lines.append(f"{VARIANTS[key]:<22}{row['rho']:>6}{k:>7}{vals}")
    table = "\n".join(table_lines)
    print(table)
    with open(os.path.join(args.out_dir, "ws3_table.txt"), "w") as f:
        f.write(table + "\n")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.2), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for key, rows in data.items():
            rhos = [r["rho"] for r in rows]
            vals = [r[metric] for r in rows]
            ax.plot(rhos, vals, marker="o", label=VARIANTS[key])
        ax.set_xlabel("compute budget $\\rho = K/K_{full}$")
        ax.set_title(metric)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "quality_vs_compute.png"), dpi=150)
    print(f"saved -> {args.out_dir}/quality_vs_compute.png")


if __name__ == "__main__":
    main()
