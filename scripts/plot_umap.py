"""
plot_umap.py
------------
Plot UMAP of [CLS] embeddings at three levels:
  input embedding, layer 1, layer 2 (final)

Usage
-----
python scripts/plot_umap.py \
    --emb_dir  experiments/logs/embeddings/Qb_vs_naive/ \
    --out      experiments/logs/embeddings/Qb_vs_naive/umap.png
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

COLORS  = {"antigen": "#EF4444", "naive": "#3B82F6"}
LEVELS  = ["emb_input", "emb_layer1", "emb_layer2"]
TITLES  = ["Input Embedding", "Layer 1 Output", "Layer 2 Output (Final)"]


def main(args):
    emb_dir = Path(args.emb_dir)
    meta    = pd.read_csv(emb_dir / "metadata.csv")

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

        for label in ["naive", "antigen"]:
            mask = meta["label"] == label
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=COLORS[label],
                label=label,
                s=2, alpha=0.3, linewidths=0, rasterized=True
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(markerscale=5, fontsize=10, framealpha=0.3,
                  handles=[
                      plt.scatter([], [], c=COLORS["antigen"], s=20, label="antigen"),
                      plt.scatter([], [], c=COLORS["naive"],   s=20, label="naive"),
                  ])

    fig.suptitle("UMAP of [CLS] representations across Transformer layers",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", required=True,
                        help="Directory containing emb_input.npy, emb_layer1.npy, emb_layer2.npy, metadata.csv")
    parser.add_argument("--out",     default="umap.png")
    args = parser.parse_args()
    main(args)