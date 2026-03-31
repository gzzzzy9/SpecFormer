"""
plot_learning_curve.py
----------------------
After all binary experiments finish, collect AUROC results
and plot learning curves for HA / Qb / RBD vs naive.

Usage
-----
python scripts/plot_learning_curve.py \
    --eval_dir  experiments/logs/binary/ \
    --splits_base data/splits/binary/ \
    --out       experiments/logs/learning_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N_CLONES_LIST = [100, 200, 500, 1000, 2000, 5000, "all"]
ANTIGENS      = ["HA", "Qb", "RBD"]
COLORS        = {"HA": "#F59E0B", "Qb": "#14B8A6", "RBD": "#8B5CF6"}


def load_results(eval_dir: str, splits_base: str) -> dict:
    """
    Expects evaluate.py to have saved a results.json with AUROC per experiment.
    Falls back to reading predictions_all.csv and computing AUROC manually.
    """
    results = {ag: {"n_clones": [], "auroc": []} for ag in ANTIGENS}

    for antigen in ANTIGENS:
        for n in N_CLONES_LIST:
            exp_name = f"{antigen}_vs_naive_{n}"
            res_path = Path(eval_dir) / exp_name / "results.json"

            if not res_path.exists():
                print(f"  [SKIP] {res_path} not found")
                continue

            with open(res_path) as f:
                res = json.load(f)

            auroc = res.get("antigen_auroc", res.get("macro_auroc"))
            if auroc is None:
                continue

            # Get actual n_clones from meta.json
            meta_path = Path(splits_base) / exp_name / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                actual_n = meta["n_clones"]
            else:
                actual_n = n if n != "all" else 99999

            results[antigen]["n_clones"].append(actual_n)
            results[antigen]["auroc"].append(auroc)

    return results


def plot(results: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for antigen in ANTIGENS:
        xs = results[antigen]["n_clones"]
        ys = results[antigen]["auroc"]
        if not xs:
            continue
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]

        ax.plot(xs, ys, "o-", color=COLORS[antigen], label=antigen,
                linewidth=2, markersize=7)

    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1, label="Random")
    ax.axhline(0.9, color="#94A3B8", linestyle=":",  linewidth=1, alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Number of antigen clones", fontsize=12)
    ax.set_ylabel("AUROC (antigen vs naive)", fontsize=12)
    ax.set_title("Learning curve: antigen-specificity detection", fontsize=13)
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",    default="experiments/logs/binary/")
    parser.add_argument("--splits_base", default="data/splits/binary/")
    parser.add_argument("--out",         default="experiments/logs/learning_curve.png")
    args = parser.parse_args()

    results = load_results(args.eval_dir, args.splits_base)
    plot(results, args.out)
