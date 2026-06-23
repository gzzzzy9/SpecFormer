"""
train_classifier_on_embeddings.py
-----------------------------------
Train a simple classification head (Logistic Regression) on frozen PLM
embeddings extracted by extract_plm_embeddings.py, and evaluate using the
same metrics as SpecFormer (Precision/Recall/F1/AUROC) for direct comparison.

Usage
-----
python scripts/train_classifier_on_embeddings.py \
    --emb_dir   experiments/logs/plm_baseline/antiberta2/ \
    --out_dir   experiments/logs/plm_baseline/antiberta2/ \
    --model_name AntiBERTa2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score)


def load_split(emb_dir: Path, split: str):
    emb = np.load(emb_dir / f"{split}_embeddings.npy")
    labels_df = pd.read_csv(emb_dir / f"{split}_labels.csv")
    labels = labels_df["label"].values
    return emb, labels


def main(args):
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    X_train, y_train = load_split(emb_dir, "train")
    X_val,   y_val   = load_split(emb_dir, "val")
    X_test,  y_test  = load_split(emb_dir, "test")

    print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    positive_label = "antigen" if "antigen" in set(y_train) else sorted(set(y_train))[0]
    print(f"  Positive class for metrics: '{positive_label}'")

    # Combine train+val for final fit (val used only for early model selection
    # if needed; here we do a simple C-sweep using val, then refit on train+val)
    best_C, best_f1 = None, -1
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        f1 = f1_score(y_val, val_pred, pos_label=positive_label)
        print(f"  C={C}: val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_C = f1, C

    print(f"\nBest C={best_C} (val F1={best_f1:.4f})")

    # Refit on train+val with best C, evaluate on test
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=best_C)
    clf.fit(X_trainval, y_trainval)

    test_pred = clf.predict(X_test)
    test_proba = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    pos_idx = classes.index(positive_label)
    test_proba_pos = test_proba[:, pos_idx]

    precision = precision_score(y_test, test_pred, pos_label=positive_label)
    recall    = recall_score(y_test, test_pred, pos_label=positive_label)
    f1        = f1_score(y_test, test_pred, pos_label=positive_label)
    y_test_binary = (y_test == positive_label).astype(int)
    auroc     = roc_auc_score(y_test_binary, test_proba_pos)

    results = {
        "model_name": args.model_name,
        "best_C": best_C,
        "test_precision": round(float(precision), 4),
        "test_recall":    round(float(recall), 4),
        "test_f1":        round(float(f1), 4),
        "test_auroc":     round(float(auroc), 4),
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
    }

    print("\n=== Test Set Results ===")
    print(f"  Model:     {args.model_name}")
    print(f"  Precision: {results['test_precision']}")
    print(f"  Recall:    {results['test_recall']}")
    print(f"  F1:        {results['test_f1']}")
    print(f"  AUROC:     {results['test_auroc']}")

    out_path = out_dir / "classifier_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", required=True,
                        help="Dir with train/val/test_embeddings.npy + labels.csv")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="PLM_baseline")
    main(parser.parse_args())
