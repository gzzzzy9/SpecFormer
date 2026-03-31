"""
evaluate.py
-----------
Load a saved checkpoint and evaluate on the test set.
Saves per-sequence predictions, confidence scores, and wrong predictions.

Usage
-----
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/small/seed42/best_model.pt \
    --config     experiments/configs/small.yaml \
    --out_dir    experiments/logs/eval/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
import pandas as pd
import numpy as np
from specformer.dataset import BCRDataset
from specformer.model import SpecFormer
from specformer.tokenizer import BCRTokenizer
from specformer.trainer import load_checkpoint

try:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    HAS_SKLEARN = True
    print('Sklearn found. Will use sklearn for evaluation metrics')
except Exception:
    HAS_SKLEARN = False
    print('Sklearn not found.')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Pure-Python AUROC (no sklearn dependency)
# ---------------------------------------------------------------------------

def binary_auroc(labels, scores, pos_class):
    """One-vs-Rest AUROC for one class."""
    binary_labels = [1 if l == pos_class else 0 for l in labels]
    paired = sorted(zip(scores, binary_labels), key=lambda x: -x[0])
    
    n_pos = sum(binary_labels)
    n_neg = len(binary_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    
    tp, fp = 0, 0
    auc = 0.0
    prev_fp = 0
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    # remaining
    auc += tp * (n_neg - prev_fp)
    return auc / (n_pos * n_neg)


def evaluate(cfg: dict, checkpoint_path: str, out_dir: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = BCRTokenizer(max_length=cfg["model"]["max_seq_len"], tag_cdrs=True)

    label_map_path = Path(cfg["data"]["processed_dir"]) / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    id2label    = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    target_names = [id2label[i] for i in range(num_classes)]

    # ---- Load test CSV directly (to get sequences) ----
    test_csv = Path(cfg["data"]["splits_dir"]) / "test.csv"
    dataset  = BCRDataset(test_csv, label_map, tok)
    loader   = torch.utils.data.DataLoader(
        dataset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["training"].get("num_workers", 4),
        pin_memory  = torch.cuda.is_available(),
    )
    test_df = pd.read_csv(test_csv)

    # ---- Model ----
    model = SpecFormer(
        num_classes = num_classes,
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

    # ---- Inference ----
    all_preds, all_labels = [], []
    all_probs = []   # (N, num_classes) softmax probabilities

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cdr_mask       = batch["cdr_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           cdr_mask=cdr_mask)

            probs = F.softmax(logits, dim=-1)   # (B, num_classes)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()  # (N, num_classes)

    # ---- Per-class report ----
    print("\n=== Classification Report (Test Set) ===")
    if HAS_SKLEARN:
        print(classification_report(all_labels, all_preds,
                                    target_names=target_names, digits=4))
        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print("=== Confusion Matrix ===")
        print(cm_df.to_string())
        auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        print(f"\nMacro AUROC: {auroc:.4f}")
    else:
        print(f"{'Class':<10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'AUROC':>8}  {'Support':>8}")
        print("-" * 62)
        aurocs = []
        for cls_idx, cls_name in enumerate(target_names):
            tp = sum(p == cls_idx and l == cls_idx for p, l in zip(all_preds, all_labels))
            fp = sum(p == cls_idx and l != cls_idx for p, l in zip(all_preds, all_labels))
            fn = sum(p != cls_idx and l == cls_idx for p, l in zip(all_preds, all_labels))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # One-vs-Rest AUROC
            scores = [all_probs[i][cls_idx] for i in range(len(all_labels))]
            auroc  = binary_auroc(all_labels, scores, cls_idx)
            aurocs.append(auroc)
            
            print(f"{cls_name:<10}  {precision:>10.4f}  {recall:>8.4f}  {f1:>8.4f}  {auroc:>8.4f}  {tp+fn:>8}")

        print(f"\nMacro AUROC: {sum(a for a in aurocs if not (a!=a)) / sum(1 for a in aurocs if not (a!=a)):.4f}")

    # ---- Build per-sequence result DataFrame ----
    rows = []
    for i in range(len(all_labels)):
        true_label = id2label[all_labels[i]]
        pred_label = id2label[all_preds[i]]
        probs_i    = all_probs[i]
        confidence = float(probs_i[all_preds[i]])   # confidence for predicted class

        row = {
            "aaSeq":        test_df["aaSeq"].iloc[i],
            "aaSeqCDR3":    test_df["aaSeqCDR3"].iloc[i] if "aaSeqCDR3" in test_df.columns else "",
            "true_label":   true_label,
            "pred_label":   pred_label,
            "correct":      true_label == pred_label,
            "confidence":   round(confidence, 4),
        }
        # Add per-class probabilities
        for cls_idx, cls_name in enumerate(target_names):
            row[f"prob_{cls_name}"] = round(float(probs_i[cls_idx]), 4)
        rows.append(row)

    result_df = pd.DataFrame(rows)

    # ---- Save all predictions ----
    all_path = out_dir / "predictions_all.csv"
    result_df.to_csv(all_path, index=False)
    print(f"\nAll predictions saved → {all_path}")

    # ---- Save wrong predictions only ----
    wrong_df = result_df[result_df["correct"] == False].sort_values("confidence", ascending=False)
    wrong_path = out_dir / "predictions_wrong.csv"
    wrong_df.to_csv(wrong_path, index=False)
    print(f"Wrong predictions saved → {wrong_path}  ({len(wrong_df)} sequences)")

    # ---- Confidence distribution summary ----
    print("\n=== Confidence Distribution ===")
    print(f"{'Class':<10}  {'Correct Mean':>12}  {'Wrong Mean':>10}  {'Correct N':>10}  {'Wrong N':>8}")
    print("-" * 58)
    for cls_name in target_names:
        cls_df  = result_df[result_df["true_label"] == cls_name]
        correct = cls_df[cls_df["correct"] == True]["confidence"]
        wrong   = cls_df[cls_df["correct"] == False]["confidence"]
        print(f"{cls_name:<10}  {correct.mean():>12.4f}  {wrong.mean() if len(wrong)>0 else float('nan'):>10.4f}"
              f"  {len(correct):>10}  {len(wrong):>8}")

    # ---- Low confidence correct predictions (borderline cases) ----
    borderline = result_df[
        (result_df["correct"] == True) & (result_df["confidence"] < 0.6)
    ].sort_values("confidence")
    borderline_path = out_dir / "predictions_borderline.csv"
    borderline.to_csv(borderline_path, index=False)
    print(f"\nBorderline correct (confidence < 0.6) → {borderline_path}  ({len(borderline)} sequences)")
    
    # ---- Save results.json (for learning curve plotting) ----
    # Compute per-class metrics for JSON
    per_class_results = {}
    aurocs_for_json = []
    for cls_idx, cls_name in enumerate(target_names):
        tp = sum(p == cls_idx and l == cls_idx for p, l in zip(all_preds, all_labels))
        fp = sum(p == cls_idx and l != cls_idx for p, l in zip(all_preds, all_labels))
        fn = sum(p != cls_idx and l == cls_idx for p, l in zip(all_preds, all_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores = [float(all_probs[i][cls_idx]) for i in range(len(all_labels))]
        auroc  = binary_auroc(all_labels, scores, cls_idx)
        aurocs_for_json.append(auroc)
        per_class_results[cls_name] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "auroc":     round(auroc, 4) if auroc == auroc else None,
            "support":   tp + fn,
        }

    valid_aurocs = [a for a in aurocs_for_json if a == a]
    macro_auroc  = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")

    # antigen_auroc: for binary experiments, the non-naive class AUROC
    antigen_cls  = next((k for k in per_class_results if k != "naive"), None)
    antigen_auroc = per_class_results[antigen_cls]["auroc"] if antigen_cls else macro_auroc

    results_json = {
        "per_class":    per_class_results,
        "macro_auroc":  round(macro_auroc, 4),
        "antigen_auroc": antigen_auroc,
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--config",      default="experiments/configs/small.yaml")
    parser.add_argument("--out_dir",     default="experiments/logs/eval/")
    parser.add_argument("--splits_dir",  default=None,
                        help="Override splits_dir in config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.splits_dir is not None:
        cfg["data"]["splits_dir"] = args.splits_dir
        print(f"splits_dir overridden to {args.splits_dir}")

    evaluate(cfg, args.checkpoint, args.out_dir)