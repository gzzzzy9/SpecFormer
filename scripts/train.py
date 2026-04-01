"""
train.py
--------
Entry point for training SpecFormer.

Usage
-----
# Local
python scripts/train.py --config experiments/configs/base.yaml

# SLURM
sbatch scripts/run_slurm.sh
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specformer.dataset import make_dataloaders
from specformer.model import SpecFormer
from specformer.tokenizer import BCRTokenizer
from specformer.trainer import Trainer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    set_seed(cfg["training"]["seed"])

    # ---- Tokenizer ----
    tok = BCRTokenizer(
        max_length=cfg["model"]["max_seq_len"],
        tag_cdrs=cfg["model"].get("use_cdr_tag", True)
    )

    # ---- DataLoaders ----
    print("=== Building DataLoaders ===")
    loaders = make_dataloaders(
        splits_dir    = cfg["data"]["splits_dir"],
        processed_dir = cfg["data"]["processed_dir"],
        tokenizer     = tok,
        batch_size    = cfg["training"]["batch_size"],
        num_workers   = cfg["training"].get("num_workers", 4),
    )

    # ---- Label map ----
    # Check for label_map in splits_dir first (for binary classification), then fallback to processed_dir
    splits_label_map = Path(cfg["data"]["splits_dir"]) / "label_map.json"
    processed_label_map = Path(cfg["data"]["processed_dir"]) / "label_map.json"

    if splits_label_map.exists():
        label_map_path = splits_label_map
        print(f"Using label_map from splits_dir: {label_map_path}")
    else:
        label_map_path = processed_label_map
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"Classes ({num_classes}): {label_map}")

    # ---- Model ----
    print("\n=== Building Model ===")
    model = SpecFormer(
        num_classes       = num_classes,
        vocab_size        = len(tok),
        d_model           = cfg["model"]["d_model"],
        n_heads           = cfg["model"]["n_heads"],
        n_layers          = cfg["model"]["n_layers"],
        d_ff              = cfg["model"]["d_ff"],
        dropout           = cfg["model"]["dropout"],
        max_seq_len       = cfg["model"]["max_seq_len"],
        use_cdr_embedding = True,
    )
    print(model)

    # ---- Trainer ----
    print("\n=== Starting Training ===")
    trainer = Trainer(
        model        = model,
        train_loader = loaders["train"],
        val_loader   = loaders["val"],
        label_map    = label_map,
        train_csv    = Path(cfg["data"]["splits_dir"]) / "train.csv",
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
        warmup_steps = cfg["training"]["warmup_steps"],
        epochs       = cfg["training"]["epochs"],
        patience     = cfg["training"]["early_stopping_patience"],
        save_dir     = cfg["logging"]["save_dir"],
    )

    history = trainer.fit()

    # ---- Save training history ----
    save_dir = Path(cfg["logging"]["save_dir"])
    history_path = save_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved → {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="experiments/configs/base.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override seed in config (for multi-seed runs)"
    )
    parser.add_argument(
        "--splits_dir", default=None,
        help="Override splits_dir in config (e.g. data/splits/clone_filtered)"
    )
    args = parser.parse_args()
    cfg  = load_config(args.config)

    # Override seed if provided via CLI
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
        cfg["logging"]["save_dir"] = f"{cfg['logging']['save_dir']}/seed{args.seed}"
        print(f"Seed overridden to {args.seed}")

    # Override splits_dir if provided via CLI
    if args.splits_dir is not None:
        cfg["data"]["splits_dir"] = args.splits_dir
        print(f"splits_dir overridden to {args.splits_dir}")

    main(cfg)