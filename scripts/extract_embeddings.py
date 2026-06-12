"""
extract_embeddings.py
---------------------
Extract [CLS] representations at three levels:
  1. Input embedding (before Transformer)
  2. Layer 1 output
  3. Layer 2 output (final)

Usage
-----
python scripts/extract_embeddings.py \
    --checkpoint experiments/checkpoints/binary/Qb_vs_naive_all/seed42/best_model.pt \
    --config     experiments/configs/small.yaml \
    --splits_dir data/splits/binary/Qb_vs_naive_all \
    --out_dir    experiments/logs/embeddings/Qb_vs_naive/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specformer.dataset import BCRDataset
from specformer.model import SpecFormer
from specformer.tokenizer import BCRTokenizer
from specformer.trainer import load_checkpoint


def extract(cfg, checkpoint_path, splits_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = BCRTokenizer(max_length=cfg["model"]["max_seq_len"], tag_cdrs=True)

    # label map
    splits_label_map    = Path(splits_dir) / "label_map.json"
    processed_label_map = Path(cfg["data"]["processed_dir"]) / "label_map.json"
    label_map_path = splits_label_map if splits_label_map.exists() else processed_label_map
    with open(label_map_path) as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    model = SpecFormer(
        num_classes = len(label_map),
        vocab_size  = len(tok),
        d_model     = cfg["model"]["d_model"],
        n_heads     = cfg["model"]["n_heads"],
        n_layers    = cfg["model"]["n_layers"],
        d_ff        = cfg["model"]["d_ff"],
        dropout     = 0.0,
        max_seq_len = cfg["model"]["max_seq_len"],
    )
    load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    # ── Hooks to capture intermediate representations ──
    input_cls_reprs  = []
    layer1_cls_reprs = []
    layer2_cls_reprs = []

    def hook_input(module, input, output):
        input_cls_reprs.append(output[:, 0, :].detach().cpu())

    def hook_layer1(module, input, output):
        layer1_cls_reprs.append(output[:, 0, :].detach().cpu())

    def hook_layer2(module, input, output):
        layer2_cls_reprs.append(output[:, 0, :].detach().cpu())

    h0 = model.embedding.register_forward_hook(hook_input)
    h1 = model.encoder.layers[0].register_forward_hook(hook_layer1)
    h2 = model.encoder.layers[-1].register_forward_hook(hook_layer2)

    # ── Run inference on all splits ──
    all_labels, all_preds = [], []

    for split in ["train", "val", "test"]:
        csv_path = Path(splits_dir) / f"{split}.csv"
        if not csv_path.exists():
            continue
        dataset = BCRDataset(csv_path, label_map, tok)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=256, shuffle=False,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        print(f"Extracting {split}: {len(dataset)} sequences...")
        for batch in loader:
            with torch.no_grad():
                logits = model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                    cdr_mask       = batch["cdr_mask"].to(device),
                )
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    h0.remove()
    h1.remove()
    h2.remove()

    # ── Save ──
    emb_input  = torch.cat(input_cls_reprs,  dim=0).numpy()
    emb_layer1 = torch.cat(layer1_cls_reprs, dim=0).numpy()
    emb_layer2 = torch.cat(layer2_cls_reprs, dim=0).numpy()

    np.save(out_dir / "emb_input.npy",  emb_input)
    np.save(out_dir / "emb_layer1.npy", emb_layer1)
    np.save(out_dir / "emb_layer2.npy", emb_layer2)

    labels_str = [id2label[l] for l in all_labels]
    preds_str  = [id2label[p] for p in all_preds]
    meta = pd.DataFrame({
        "label":   labels_str,
        "pred":    preds_str,
        "correct": [l == p for l, p in zip(labels_str, preds_str)],
    })
    meta.to_csv(out_dir / "metadata.csv", index=False)

    print(f"\nSaved embeddings → {out_dir}")
    print(f"  emb_input:  {emb_input.shape}")
    print(f"  emb_layer1: {emb_layer1.shape}")
    print(f"  emb_layer2: {emb_layer2.shape}")
    print(f"  metadata:   {len(meta)} sequences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--config",      default="experiments/configs/small.yaml")
    parser.add_argument("--splits_dir",  required=True)
    parser.add_argument("--out_dir",     default="experiments/logs/embeddings/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    extract(cfg, args.checkpoint, args.splits_dir, args.out_dir)