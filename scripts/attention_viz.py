"""
attention_viz.py
----------------
Extract and visualize attention weights from SpecFormer.

Usage
-----
python scripts/attention_viz.py \
    --checkpoint experiments/checkpoints/small/seed42/best_model.pt \
    --config     experiments/configs/small.yaml \
    --n_samples  30 \
    --out_dir    experiments/logs/attention/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specformer.dataset import BCRDataset
from specformer.model import SpecFormer
from specformer.tokenizer import BCRTokenizer
from specformer.trainer import load_checkpoint

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 获取字体文件的绝对路径
font_path = os.path.abspath('fonts/helvetica-255/Helvetica.ttf')

# 检查文件是否存在
if os.path.exists(font_path):
    # 核心：直接把这个路径注册到 Matplotlib
    fm.fontManager.addfont(font_path)
    # 设置为默认字体
    plt.rcParams['font.sans-serif'] = ['Helvetica']
else:
    print(f"警告：找不到字体文件 {font_path}，将使用系统默认字体")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cdr_spans(tokens: list) -> dict:
    spans = {}
    for cdr in ["<cdr1>", "<cdr2>", "<cdr3>"]:
        positions = [i for i, t in enumerate(tokens) if t == cdr]
        if len(positions) == 2:
            spans[cdr] = (positions[0], positions[1])
    return spans


def compute_cls_attn(attn_weights: list, real_len: int, tokens: list) -> np.ndarray:
    stacked = torch.stack([a.squeeze(0) for a in attn_weights], dim=0)
    mean_attn = stacked.mean(dim=(0, 1))
    cls_row = mean_attn[0, :real_len].float().numpy().copy()  # 加 .copy()

    # 找出非 special token 的位置
    special = {"<cls>", "<eos>", "<pad>", "<mask>"}
    valid_positions = [i for i, t in enumerate(tokens) if t not in special]

    # 只在 valid positions 上归一化
    valid_vals = np.array([cls_row[i] for i in valid_positions], dtype=np.float64)
    total = float(valid_vals.sum()) + 1e-8
    valid_vals = valid_vals / total

    # 把归一化后的值放回
    result = np.zeros(real_len, dtype=np.float64)
    for i, pos in enumerate(valid_positions):
        result[pos] = float(valid_vals[i])

    return result


def classify_tokens(tokens: list, cdr_spans: dict) -> list:
    """Return color for each token position."""
    cdr_positions = set()
    for s, e in cdr_spans.values():
        cdr_positions.update(range(s + 1, e))

    colors = []
    for i, tok in enumerate(tokens):
        if tok in ("<cls>", "<eos>", "<pad>"):
            colors.append("#94A3B8")
        elif tok in ("<cdr1>", "<cdr2>", "<cdr3>"):
            colors.append("#F59E0B")
        elif i in cdr_positions:
            colors.append("#EF4444")
        else:
            colors.append("#3B82F6")
    return colors


# ---------------------------------------------------------------------------
# Plot 1: CLS attention bar chart
# ---------------------------------------------------------------------------

def plot_cls_attention(tokens, attn_weights, cdr_spans, label, seq_id, out_path):
    real_len = len(tokens)
    cls_attn = compute_cls_attn(attn_weights, real_len, tokens)   # numpy (real_len,)
    colors   = classify_tokens(tokens, cdr_spans)

    # CDR vs FR means
    cdr_positions = set()
    for s, e in cdr_spans.values():
        cdr_positions.update(range(s + 1, e))
    fr_positions = [
        i for i, t in enumerate(tokens)
        if t not in ("<cls>", "<eos>", "<pad>", "<cdr1>", "<cdr2>", "<cdr3>")
        and i not in cdr_positions
    ]
    cdr_mean = float(cls_attn[sorted(cdr_positions)].mean()) if cdr_positions else 0.0
    fr_mean  = float(cls_attn[fr_positions].mean())          if fr_positions  else 0.0

    fig = plt.figure(figsize=(max(14, real_len * 0.28), 7))

    # ── Top panel: bar chart ──
    fig = plt.figure(figsize=(max(14, real_len * 0.28), 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x_vals    = list(range(real_len))
    y_vals    = [float(v) for v in cls_attn]

    for xi, (yi, ci) in enumerate(zip(y_vals, colors)):
        ax1.bar(xi, yi, color=ci, width=0.8, linewidth=0)

    ax1.set_xlim(-0.5, real_len - 0.5)
    ax1.set_ylim(0, max(y_vals) * 1.15 + 1e-8)
    ax1.set_ylabel("Attention weight", fontsize=10)
    ax1.set_title(
        f"[CLS] Attention  |  label={label}  |  {seq_id}",
        fontsize=11, pad=8
    )
    ax1.tick_params(colors="black", labelsize=7)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#CCCCCC")

    # x ticks
    # step = max(1, real_len // 30)
    step = 1
    ax1.set_xticks(x_vals[::step])
    ax1.set_xticklabels(tokens[::step], fontsize=6, rotation=0)

    # CDR region shading
    drawn_label = False
    for s, e in cdr_spans.values():
        ax1.axvspan(s, e, alpha=0.12, color="#EF4444",
                    label="CDR region" if not drawn_label else "")
        drawn_label = True

    legend_elements = [
        mpatches.Patch(color="#3B82F6", label="Framework"),
        mpatches.Patch(color="#EF4444", label="CDR region"),
        mpatches.Patch(color="#F59E0B", label="CDR marker"),
        mpatches.Patch(color="#94A3B8", label="Special token"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right",
               framealpha=0.3, labelcolor="black", fontsize=8)

    # ── Bottom panel: CDR vs FR bar ──
    ax2.bar([0, 1], [fr_mean, cdr_mean], color=["#3B82F6", "#EF4444"],
            width=0.4, linewidth=0)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["FR", "CDR"], fontsize=9)
    ax2.set_title("Mean attention", fontsize=11)
    ax2.tick_params(colors="black", labelsize=8)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#CCCCCC")

    plt.savefig(str(out_path), dpi=500, bbox_inches="tight")
    plt.close("all")

    return cdr_mean, fr_mean


# ---------------------------------------------------------------------------
# Plot 2: Layer heatmap
# ---------------------------------------------------------------------------

def plot_layer_heatmap(tokens, attn_weights, label, seq_id, out_path):
    real_len  = len(tokens)
    n_layers  = len(attn_weights)

    layer_attn = np.zeros((n_layers, real_len))
    for li, attn in enumerate(attn_weights):
        a = attn.squeeze(0).float()            # (n_heads, L, L)
        cls_row = a[:, 0, :real_len]           # (n_heads, real_len)
        layer_attn[li] = cls_row.mean(0).numpy()

    fig, ax = plt.subplots(figsize=(max(10, real_len * 0.25), n_layers * 0.9 + 2))

    im = ax.imshow(layer_attn, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i+1}" for i in range(n_layers)], fontsize=9)

    step = 1
    ax.set_xticks(list(range(0, real_len, step)))
    ax.set_xticklabels(tokens[::step], rotation=0, fontsize=6)
    ax.set_title(f"Layer-wise attention  |  label={label}  |  {seq_id}", fontsize=11, pad=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(colors="black", labelsize=8)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=500, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# Plot 3: Summary CDR vs FR per class
# ---------------------------------------------------------------------------

def plot_summary(results: list, out_path: Path) -> None:
    """
    Two-panel summary:
    Top:    CDR (total) vs Framework mean attention per class
    Bottom: CDR1 / CDR2 / CDR3 breakdown per class
    """
    class_data = defaultdict(lambda: {
        "cdr": [], "fr": [], "cdr1": [], "cdr2": [], "cdr3": []
    })
    for r in results:
        class_data[r["label"]]["cdr"].append(r["cdr_mean"])
        class_data[r["label"]]["fr"].append(r["fr_mean"])
        class_data[r["label"]]["cdr1"].append(r["cdr1_mean"])
        class_data[r["label"]]["cdr2"].append(r["cdr2_mean"])
        class_data[r["label"]]["cdr3"].append(r["cdr3_mean"])

    labels = sorted(class_data.keys())
    n      = len(labels)
    x      = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9))

    # ── Panel 1: CDR total vs Framework ──────────────────────────
    w = 0.35
    fr_m   = [np.mean(class_data[l]["fr"])  for l in labels]
    cdr_m  = [np.mean(class_data[l]["cdr"]) for l in labels]
    fr_s   = [np.std(class_data[l]["fr"])   for l in labels]
    cdr_s  = [np.std(class_data[l]["cdr"])  for l in labels]

    ax1.bar(x - w/2, fr_m,  w, label="Framework",   color="#3B82F6",
            yerr=fr_s,  capsize=4, error_kw={"ecolor": "white", "capthick": 1.5})
    ax1.bar(x + w/2, cdr_m, w, label="CDR regions", color="#EF4444",
            yerr=cdr_s, capsize=4, error_kw={"ecolor": "white", "capthick": 1.5})

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel("Mean [CLS] attention", fontsize=10)
    ax1.set_title("CDR vs FR attention summary", fontsize=12, pad=8)
    ax1.tick_params(colors="black")
    ax1.legend(labelcolor="black", framealpha=0.3, fontsize=9)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#CCCCCC")

    # ── Panel 2: CDR1 / CDR2 / CDR3 breakdown ────────────────────
    w3     = 0.22
    colors = {"cdr1": "#60A5FA", "cdr2": "#34D399", "cdr3": "#F87171"}
    labels_cdr = ["CDR1", "CDR2", "CDR3"]
    offsets    = [-w3, 0, w3]

    for (cdr_key, color, label_cdr, offset) in zip(
        ["cdr1", "cdr2", "cdr3"], colors.values(), labels_cdr, offsets
    ):
        means = [np.mean(class_data[l][cdr_key]) for l in labels]
        stds  = [np.std(class_data[l][cdr_key])  for l in labels]
        ax2.bar(x + offset, means, w3, label=label_cdr, color=color,
                yerr=stds, capsize=3,
                error_kw={"ecolor": "white", "capthick": 1.2})

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylabel("Mean [CLS] attention", fontsize=10)
    ax2.set_title("CDR1 / CDR2 / CDR3 attention breakdown by antigen class", fontsize=12, pad=8)
    ax2.tick_params(colors="black")
    ax2.legend(labelcolor="black", framealpha=0.3, fontsize=9)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#CCCCCC")

    plt.tight_layout(pad=2.0)
    plt.savefig(str(out_path), dpi=500, bbox_inches="tight")
    plt.close("all")
    print(f"Saved summary → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tok = BCRTokenizer(max_length=cfg["model"]["max_seq_len"], tag_cdrs=True)

    with open(Path(cfg["data"]["processed_dir"]) / "label_map.json") as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    csv_path = Path(cfg["data"]["splits_dir"]) / f"{args.split}.csv"
    dataset  = BCRDataset(csv_path, label_map, tok)
    print(f"Loaded {len(dataset)} sequences from {args.split} split")

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
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    # Sample n_per_class sequences per label
    label_to_idx = defaultdict(list)
    for i in range(len(dataset)):
        label_to_idx[dataset.labels[i]].append(i)

    n_per_class = max(1, args.n_samples // len(label_map))
    selected = []
    rng = np.random.default_rng(42)
    for label_idx, indices in label_to_idx.items():
        chosen = rng.choice(indices, size=min(n_per_class, len(indices)),
                            replace=False)
        selected.extend([(int(i), label_idx) for i in chosen])

    print(f"Visualizing {len(selected)} sequences ({n_per_class} per class)...")

    summary = []

    for si, (idx, true_label) in enumerate(selected):
        sample = dataset[idx]
        input_ids      = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        cdr_mask       = sample["cdr_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits, attn_weights = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cdr_mask=cdr_mask,
                return_attentions=True,
            )
        # attn_weights: list of (1, n_heads, L, L) CPU tensors

        pred_label = int(logits.argmax(-1).item())
        real_len   = int(attention_mask[0].sum().item())
        label_str  = id2label[true_label]
        pred_str   = id2label[pred_label]
        correct    = "correct" if true_label == pred_label else "wrong"

        # Build token list (strings)
        tok_ids  = sample["input_ids"][:real_len].tolist()
        tok_list = [tok.id2tok.get(i, "?") for i in tok_ids]

        cdr_spans = get_cdr_spans(tok_list)
        seq_id    = f"idx{idx}_{correct}_pred{pred_str}"

        cls_path   = out_dir / f"{label_str}_{si:03d}_cls.png"
        layer_path = out_dir / f"{label_str}_{si:03d}_layers.png"

        cdr_mean, fr_mean = plot_cls_attention(
            tok_list, attn_weights, cdr_spans, label_str, seq_id, cls_path
        )
        plot_layer_heatmap(
            tok_list, attn_weights, label_str, seq_id, layer_path
        )

        # Per-CDR attention
        cls_attn = compute_cls_attn(attn_weights, real_len, tok_list)
        cdr_detail = {}
        for cdr_name, (s, e) in cdr_spans.items():
            positions = list(range(s + 1, e))
            cdr_detail[cdr_name] = float(cls_attn[positions].mean()) if positions else 0.0
            
        marker_positions = [i for i, t in enumerate(tok_list) 
                    if t in ("<cdr1>", "<cdr2>", "<cdr3>")]
        marker_attn = float(np.mean([cls_attn[p] for p in marker_positions])) \
                    if marker_positions else 0.0
        
        cdr_internal = []
        for s, e in cdr_spans.items():
            cdr_internal.extend(range(cdr_spans[s][0] + 1, cdr_spans[s][1]))
        cdr_internal_attn = float(np.mean([cls_attn[p] for p in cdr_internal])) \
                            if cdr_internal else 0.0

        summary.append({
            "label":    label_str,
            "cdr_mean": cdr_mean,
            "fr_mean":  fr_mean,
            "cdr1_mean": cdr_detail.get("<cdr1>", 0.0),
            "cdr2_mean": cdr_detail.get("<cdr2>", 0.0),
            "cdr3_mean": cdr_detail.get("<cdr3>", 0.0),
            "correct":  true_label == pred_label,
            "marker_attn":      marker_attn,       
            "cdr_internal_attn": cdr_internal_attn, 
        })

        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{len(selected)} done")

    plot_summary(summary, out_dir / "summary_cdr_vs_fr.png")
    pd.DataFrame(summary).to_csv(out_dir / "attention_summary.csv", index=False)
    print(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="experiments/configs/small.yaml")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n_samples",  type=int, default=30)
    parser.add_argument("--out_dir",    default="experiments/logs/attention/")
    main(parser.parse_args())