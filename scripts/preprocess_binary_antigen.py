"""
preprocess_binary_antigen_v2.py
--------------------------------
Antigen vs antigen binary classification (e.g. Qb vs RBD) with MMseqs2-based
clonotype definition. Adapted from preprocess_binary_antigen.py.

Clonotype = same IGHV gene (allele-stripped) + CDR3 >=80% identity, same length.
MMseqs2 is called ONCE for all sequences, then V gene filtering is applied in Python.

Usage
-----
python scripts/preprocess_binary_antigen_v2.py \
    --processed_dir  data/processed/ \
    --antigen1       Qb \
    --antigen2       RBD \
    --out_dir        data/splits/binary/Qb_vs_RBD_all \
    --seed           42
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
MMSEQS2_ID  = 0.8
MMSEQS2_COV = 1.0


def assign_clone_id_mmseqs2(df: pd.DataFrame,
                             identity: float = MMSEQS2_ID,
                             coverage: float = MMSEQS2_COV,
                             threads: int = 8) -> pd.DataFrame:
    """
    Call MMseqs2 ONCE on all CDR3 sequences, then enforce V gene constraint in Python.
    Two sequences are in the same clone only if:
      1. Same IGHV gene (allele-stripped)
      2. In the same MMseqs2 cluster (>=80% CDR3 identity, same length via -c 1.0)
    """
    df = df.copy()
    df["_v"]    = df["bestVHit"].str.split("*").str[0]
    df["_cdr3"] = df["aaSeqCDR3"].fillna("")
    df["_len"]  = df["_cdr3"].str.len()
    df["_rowid"] = range(len(df))   # stable integer index for FASTA names

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ── Step 1: write all CDR3 sequences to one FASTA ──
        # Header encodes rowid and V gene so we can recover them after clustering
        fasta_path = td / "all_cdr3.fasta"
        with open(fasta_path, "w") as fh:
            for _, row in df.iterrows():
                if row["_len"] == 0:
                    continue
                # Use rowid as FASTA name; V gene stored separately
                fh.write(f">{row['_rowid']}\n{row['_cdr3']}\n")

        # ── Step 2: single MMseqs2 call ──
        out_prefix = td / "cluster"
        tmp_dir    = td / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        cmd = [
            "mmseqs", "easy-cluster",
            str(fasta_path), str(out_prefix), str(tmp_dir),
            "--min-seq-id", str(identity),
            "-c",           str(coverage),
            "--cov-mode",   "3",   # coverage of shorter seq → enforces equal length
            "--threads",    str(threads),
            "-v",           "0",
        ]
        print(f"  Running MMseqs2 on {len(df)} sequences...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MMseqs2 failed:\n{result.stderr}")
        print(f"  MMseqs2 done.")

        # ── Step 3: parse cluster TSV (rep \t member) ──
        cluster_tsv = Path(str(out_prefix) + "_cluster.tsv")
        rowid_to_mmseqs_rep = {}   # rowid → MMseqs2 representative rowid

        with open(cluster_tsv) as fh:
            for line in fh:
                rep, member = line.strip().split("\t")
                rowid_to_mmseqs_rep[int(member)] = int(rep)

    # ── Step 4: apply V gene constraint in Python ──
    # Final clone_id = (V_gene, MMseqs2_rep) only if same V gene
    # Sequences with empty CDR3 get their own unique clone
    rowid_to_v = df.set_index("_rowid")["_v"].to_dict()

    global_counter = [0]
    pair_to_clone  = {}   # (v_gene, mmseqs_rep) → clone_id string

    clone_id_list = []
    for _, row in df.iterrows():
        rid = row["_rowid"]
        v   = row["_v"]

        if row["_len"] == 0:
            clone_id_list.append(f"EMPTY_{global_counter[0]}")
            global_counter[0] += 1
            continue

        mmseqs_rep = rowid_to_mmseqs_rep.get(rid, rid)
        rep_v      = rowid_to_v.get(mmseqs_rep, v)

        if rep_v == v:
            # Same V gene as representative → same clone
            key = (v, mmseqs_rep)
        else:
            # Different V gene → treat as its own cluster
            key = (v, rid)

        if key not in pair_to_clone:
            pair_to_clone[key] = f"{v}_c{global_counter[0]}"
            global_counter[0] += 1

        clone_id_list.append(pair_to_clone[key])

    df["clone_id"] = clone_id_list
    return df.drop(columns=["_v", "_cdr3", "_len", "_rowid"])


def split_by_clone(df: pd.DataFrame, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
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
        n      = len(clones)
        n_test = max(1, round(n * TEST_RATIO))
        n_val  = max(1, round(n * VAL_RATIO))
        test_clones.extend(clones[:n_test])
        val_clones.extend(clones[n_test:n_test + n_val])
        train_clones.extend(clones[n_test + n_val:])

    train_set = set(train_clones)
    val_set   = set(val_clones)
    test_set  = set(test_clones)

    assert len(train_set & val_set)  == 0
    assert len(train_set & test_set) == 0
    assert len(val_set   & test_set) == 0

    train = df[df["clone_id"].isin(train_set)].reset_index(drop=True)
    val   = df[df["clone_id"].isin(val_set)].reset_index(drop=True)
    test  = df[df["clone_id"].isin(test_set)].reset_index(drop=True)

    # Sanity check
    train_cdr3 = set(train["aaSeqCDR3"].dropna())
    test_cdr3  = set(test["aaSeqCDR3"].dropna())
    overlap    = len(train_cdr3 & test_cdr3)
    if overlap > 0:
        print(f"  [WARN] {overlap} exact CDR3 sequences appear in both train and test")

    return train, val, test


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_df = pd.read_csv(Path(args.processed_dir) / "sequences.csv")

    # ---- Antigen 1 ----
    antigen1_df = all_df[all_df["Specificity"] == args.antigen1].copy()
    print(f"=== Assigning clonotypes (MMseqs2, >=80% CDR3 identity): {args.antigen1} ===")
    antigen1_df = assign_clone_id_mmseqs2(antigen1_df)

    antigen1_clones = antigen1_df["clone_id"].unique()
    if args.n_clones is not None and args.n_clones < len(antigen1_clones):
        chosen = rng.choice(antigen1_clones, size=args.n_clones, replace=False)
        antigen1_df = antigen1_df[antigen1_df["clone_id"].isin(chosen)]
        print(f"Subsampled to {args.n_clones} clones ({len(antigen1_df)} seqs) from {args.antigen1}")
    else:
        print(f"Using all {len(antigen1_clones)} clones ({len(antigen1_df)} seqs) from {args.antigen1}")

    # ---- Antigen 2 ----
    antigen2_df = all_df[all_df["Specificity"] == args.antigen2].copy()
    print(f"\n=== Assigning clonotypes (MMseqs2, >=80% CDR3 identity): {args.antigen2} ===")
    antigen2_df = assign_clone_id_mmseqs2(antigen2_df)

    antigen2_clones = antigen2_df["clone_id"].unique()
    if args.n_clones is not None and args.n_clones < len(antigen2_clones):
        chosen = rng.choice(antigen2_clones, size=args.n_clones, replace=False)
        antigen2_df = antigen2_df[antigen2_df["clone_id"].isin(chosen)]
        print(f"Subsampled to {args.n_clones} clones ({len(antigen2_df)} seqs) from {args.antigen2}")
    else:
        print(f"Using all {len(antigen2_clones)} clones ({len(antigen2_df)} seqs) from {args.antigen2}")

    # Remove clone_id overlap between the two antigens
    antigen1_clone_ids = set(antigen1_df["clone_id"])
    num_before        = len(antigen2_df)
    antigen2_df       = antigen2_df[~antigen2_df["clone_id"].isin(antigen1_clone_ids)]
    print(f"Removed {num_before - len(antigen2_df)} {args.antigen2} seqs with clone_id overlap")
    print(f"Using all {antigen2_df['clone_id'].nunique()} clones ({len(antigen2_df)} seqs) from {args.antigen2}")
    # Keep original Specificity label (e.g. "RBD")

    # ---- Combine and split ----
    combined = pd.concat([antigen1_df, antigen2_df], ignore_index=True)
    print(f"\nCombined: {len(combined)} sequences")
    print(combined["Specificity"].value_counts().to_string())

    print(f"\n=== Splitting (cluster-aware, seed={args.seed}) ===")
    train, val, test = split_by_clone(combined, seed=args.seed)

    print(f"  train: {len(train)}  {train['Specificity'].value_counts().to_dict()}")
    print(f"  val:   {len(val)}    {val['Specificity'].value_counts().to_dict()}")
    print(f"  test:  {len(test)}   {test['Specificity'].value_counts().to_dict()}")

    # ---- Save ----
    label_map = {args.antigen1: 0, args.antigen2: 1}
    with open(Path(args.out_dir) / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    train.to_csv(Path(args.out_dir) / "train.csv", index=False)
    val.to_csv(  Path(args.out_dir) / "val.csv",   index=False)
    test.to_csv( Path(args.out_dir) / "test.csv",  index=False)

    meta = {
        "antigen1":           args.antigen1,
        "antigen2":           args.antigen2,
        "n_clones_antigen1":  int(antigen1_df["clone_id"].nunique()),
        "n_clones_antigen2":  int(antigen2_df["clone_id"].nunique()),
        "n_seqs_antigen1":    len(antigen1_df),
        "n_seqs_antigen2":    len(antigen2_df),
        "clonotype_method":   "MMseqs2",
        "mmseqs2_identity":   MMSEQS2_ID,
        "seed":               args.seed,
    }
    with open(Path(args.out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed/")
    parser.add_argument("--antigen1", required=True, choices=["HA", "Qb", "RBD"],
                        help="First antigen (positive class)")
    parser.add_argument("--antigen2", required=True, choices=["HA", "Qb", "RBD"],
                        help="Second antigen (negative class, label='naive' for train.py compat)")
    parser.add_argument("--n_clones", type=int, default=None)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out_dir",  required=True)
    main(parser.parse_args())
