"""
preprocess_binary.py
--------------------
Build binary classification datasets (one antigen vs naive B).
Supports subsampling by n_clones for learning curve experiments.

Usage
-----
# Full dataset, RBD vs naive
python scripts/preprocess_binary.py \
    --processed_dir  data/processed/ \
    --antigen        RBD \
    --out_dir        data/splits/binary/RBD_vs_naive_all

# Subsample 500 clones
python scripts/preprocess_binary.py \
    --processed_dir  data/processed/ \
    --antigen        RBD \
    --n_clones       500 \
    --out_dir        data/splits/binary/RBD_vs_naive_500

Input files required (from preprocess.py output)
-------------------------------------------------
data/processed/sequences.csv      -- antigen-specific sequences (with clone_id)
data/processed/naive_sequences.csv -- naive B sequences
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED       = 42
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_clone_id(df: pd.DataFrame) -> pd.DataFrame:
    v_gene = df["bestVHit"].str.split("*").str[0]
    j_gene = df["bestJHit"].str.split("*").str[0]
    cdr3   = df["aaSeqCDR3"].fillna("")
    df = df.copy()
    df["clone_id"] = v_gene + "_" + j_gene + "_" + cdr3
    return df


def split_by_clone(df: pd.DataFrame) -> tuple:
    rng = np.random.default_rng(SEED)

    clone_label = (
        df.groupby("clone_id")["Specificity"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"Specificity": "clone_label"})
    )

    train_clones, val_clones, test_clones = [], [], []
    for lbl, grp in clone_label.groupby("clone_label"):
        clones = grp["clone_id"].values.copy()
        rng.shuffle(clones)
        n       = len(clones)
        n_test  = max(1, round(n * TEST_RATIO))
        n_val   = max(1, round(n * VAL_RATIO))
        test_clones.extend(clones[:n_test])
        val_clones.extend(clones[n_test:n_test + n_val])
        train_clones.extend(clones[n_test + n_val:])

    train_set = set(train_clones)
    val_set   = set(val_clones)
    test_set  = set(test_clones)

    train = df[df["clone_id"].isin(train_set)]
    val   = df[df["clone_id"].isin(val_set)]
    test  = df[df["clone_id"].isin(test_set)]

    assert len(train_set & val_set)  == 0
    assert len(train_set & test_set) == 0
    assert len(val_set   & test_set) == 0

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- Load antigen-specific sequences ----
    antigen_df = pd.read_csv(Path(args.processed_dir) / "sequences.csv")

    # Filter to target antigen
    antigen_df = antigen_df[antigen_df["Specificity"] == args.antigen].copy()
    antigen_df = assign_clone_id(antigen_df)

    # Subsample by n_clones if specified
    all_clones = antigen_df["clone_id"].unique()
    if args.n_clones is not None and args.n_clones < len(all_clones):
        chosen_clones = rng.choice(all_clones, size=args.n_clones, replace=False)
        antigen_df = antigen_df[antigen_df["clone_id"].isin(chosen_clones)]
        print(f"Subsampled to {args.n_clones} clones "
              f"({len(antigen_df)} sequences) from {args.antigen}")
    else:
        print(f"Using all {len(all_clones)} clones "
              f"({len(antigen_df)} sequences) from {args.antigen}")

    antigen_df["Specificity"] = "antigen"   # binary label

    # ---- Load naive sequences ----
    naive_path = Path(args.processed_dir) / "naive_sequences.csv"
    naive_df   = pd.read_csv(naive_path)
    naive_df   = assign_clone_id(naive_df)

    # Remove naive clones overlapping with antigen clones
    antigen_clone_ids = set(antigen_df["clone_id"])
    before = len(naive_df)
    naive_df = naive_df[~naive_df["clone_id"].isin(antigen_clone_ids)]
    print(f"Removed {before - len(naive_df)} naive seqs overlapping with {args.antigen} clones")

    if args.balance_naive:
        n_antigen_clones = antigen_df["clone_id"].nunique()
        naive_clones     = naive_df["clone_id"].unique()
        if len(naive_clones) > n_antigen_clones:
            chosen_naive = rng.choice(naive_clones, size=n_antigen_clones, replace=False)
            naive_df = naive_df[naive_df["clone_id"].isin(chosen_naive)]
            print(f"Subsampled naive to {n_antigen_clones} clones ({len(naive_df)} sequences) for 1:1 balance")
        else:
            print(f"Using all {len(naive_clones)} naive clones ({len(naive_df)} sequences)")
    else:
        print(f"Using all {naive_df['clone_id'].nunique()} naive clones ({len(naive_df)} sequences)")

    naive_df["Specificity"] = "naive"

    # ---- Combine and split ----
    combined = pd.concat([antigen_df, naive_df], ignore_index=True)
    print(f"\nCombined: {len(combined)} sequences")
    print(combined["Specificity"].value_counts().to_string())

    train, val, test = split_by_clone(combined)

    print(f"\nSplit:")
    print(f"  train: {len(train)}  {train['Specificity'].value_counts().to_dict()}")
    print(f"  val:   {len(val)}    {val['Specificity'].value_counts().to_dict()}")
    print(f"  test:  {len(test)}   {test['Specificity'].value_counts().to_dict()}")

    # ---- Save ----
    label_map = {"antigen": 0, "naive": 1}
    with open(Path(args.out_dir) / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    train.to_csv(Path(args.out_dir) / "train.csv", index=False)
    val.to_csv(  Path(args.out_dir) / "val.csv",   index=False)
    test.to_csv( Path(args.out_dir) / "test.csv",  index=False)

    # Save metadata
    meta = {
        "antigen":   args.antigen,
        "n_clones":  int(antigen_df["clone_id"].nunique()),
        "n_seqs":    len(antigen_df),
    }
    with open(Path(args.out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed/")
    parser.add_argument("--antigen",       required=True,
                        choices=["HA", "Qb", "RBD"],
                        help="Target antigen for binary classification")
    parser.add_argument("--n_clones",      type=int, default=None,
                        help="Number of antigen clones to use (None = all)")
    parser.add_argument("--out_dir",       required=True,
                        help="Output directory for splits")
    parser.add_argument("--balance_naive", action="store_true",
                        help="Downsample naive to match antigen clone count (1:1). "
                             "Default: use all naive sequences.")
    main(parser.parse_args())