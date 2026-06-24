"""
plot_umap.py
------------
Plot UMAP of [CLS] embeddings at three levels:
  input embedding, layer 1, layer 2 (final)
Class names and colors are inferred automatically from metadata.csv.

Usage
-----
python scripts/plot_umap.py \
    --emb_dir  experiments/logs/embeddings/Qb_vs_RBD/seed42 \
    --out      experiments/logs/figures/umap_Qb_vs_RBD.png \
    --title    "Qb vs RBD"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from umap import UMAP

# Default color palette — extended to support any number of classes
DEFAULT_PALETTE = [
    "#EF4444", "#3B82F6", "#10B981", "#F59E0B",
    "#8B5CF6", "#EC4899", "#14B8A6", "#F97316",
]

LEVELS = ["emb_input", "emb_layer1", "emb_layer2"]
TITLES = ["Input Embedding", "Layer 1 Output", "Layer 2 Output (Final)"]


def main(args):
    emb_dir = Path(args.emb_dir)
    meta    = pd.read_csv(emb_dir / "metadata.csv")

    # ── Infer class names and assign colors ──
    classes = sorted(meta["label"].unique())
    colors  = {cls: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
               for i, cls in enumerate(classes)}
    print(f"Classes found: {classes}")
    print(f"Color map: {colors}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, level, title in zip(axes, LEVELS, TITLES):
        emb_path = emb_dir / f"{level}.npy"
        if not emb_path.exists():
            print(f"[SKIP] {emb_path} not found")
            ax.set_visible(False)
            continue

        emb = np.load(emb_path)
        print(f"Running UMAP on {level}: {emb.shape}...")

        reducer = UMAP(n_components=2, random_state=42,
                       n_neighbors=30, min_dist=0.1)
        coords  = reducer.fit_transform(emb)

        # Plot each class
        for cls in classes:
            mask = meta["label"] == cls
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors[cls],
                label=cls,
                s=2, alpha=0.3, linewidths=0, rasterized=True,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend with larger markers
        handles = [
            plt.scatter([], [], c=colors[cls], s=30, label=cls)
            for cls in classes
        ]
        ax.legend(handles=handles, markerscale=3,
                  fontsize=10, framealpha=0.3)

    suptitle = args.title if args.title else \
        f"UMAP of [CLS] representations — {emb_dir.parent.name}"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out",     required=True)
    parser.add_argument("--title",   default=None,
                        help="Plot title (default: inferred from directory name)")
    main(parser.parse_args())