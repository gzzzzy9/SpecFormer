"""
preprocess_binary_antigen_v2.py
--------------------------------
Antigen vs antigen binary classification (e.g. Qb vs RBD) with MMseqs2-based
clonotype definition. Adapted from preprocess_binary_antigen.py.

Clonotype = same IGHV gene (allele-stripped) + CDR3 >=80% identity, same length.

Usage
-----
python scripts/preprocess_binary_antigen_v2.py \
    --processed_dir  data/processed/ \
    --antigen        Qb \
    --negative       RBD \
    --out_dir        data/splits/binary_v2/Qb_vs_RBD_all \
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
    df = df.copy()
    v_gene = df["bestVHit"].str.split("*").str[0]
    cdr3   = df["aaSeqCDR3"].fillna("")
    df["_v"]    = v_gene
    df["_cdr3"] = cdr3
    df["_len"]  = cdr3.str.len()

    clone_ids = pd.Series([""] * len(df), index=df.index)
    global_counter = [0]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        for (v, l), grp in df.groupby(["_v", "_len"]):
            n = len(grp)

            if l == 0:
                for idx in grp.index:
                    clone_ids[idx] = f"EMPTY_{global_counter[0]}"
                    global_counter[0] += 1
                continue

            if n == 1:
                clone_ids[grp.index[0]] = f"{v}_l{l}_c{global_counter[0]}"
                global_counter[0] += 1
                continue

            fasta_path = td / "input.fasta"
            with open(fasta_path, "w") as fh:
                for i, (orig_idx, row) in enumerate(grp.iterrows()):
                    fh.write(f">{i}\n{row['_cdr3']}\n")

            out_prefix = td / "cluster"
            tmp_dir    = td / "tmp"
            tmp_dir.mkdir(exist_ok=True)

            cmd = [
                "mmseqs", "easy-cluster",
                str(fasta_path), str(out_prefix), str(tmp_dir),
                "--min-seq-id", str(identity),
                "-c",           str(coverage),
                "--cov-mode",   "3",
                "--threads",    str(threads),
                "-v",           "0",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"MMseqs2 failed (V={v}, L={l}):\n{result.stderr}")

            cluster_tsv = Path(str(out_prefix) + "_cluster.tsv")
            rep_to_cid  = {}
            row_indices = list(grp.index)

            with open(cluster_tsv) as fh:
                for line in fh:
                    rep, member = line.strip().split("\t")
                    if rep not in rep_to_cid:
                        rep_to_cid[rep] = global_counter[0]
                        global_counter[0] += 1
                    member_orig_idx = row_indices[int(member)]
                    clone_ids[member_orig_idx] = f"{v}_l{l}_c{rep_to_cid[rep]}"

            for f in tmp_dir.glob("*"):
                try: f.unlink()
                except: pass

    df["clone_id"] = clone_ids
    return df.drop(columns=["_v", "_cdr3", "_len"])


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

    # ---- Positive class (antigen) ----
    antigen_df = all_df[all_df["Specificity"] == args.antigen].copy()
    print(f"=== Assigning clonotypes (MMseqs2, >=80% CDR3 identity): {args.antigen} ===")
    antigen_df = assign_clone_id_mmseqs2(antigen_df)

    all_antigen_clones = antigen_df["clone_id"].unique()
    if args.n_clones is not None and args.n_clones < len(all_antigen_clones):
        chosen = rng.choice(all_antigen_clones, size=args.n_clones, replace=False)
        antigen_df = antigen_df[antigen_df["clone_id"].isin(chosen)]
        print(f"Subsampled to {args.n_clones} clones ({len(antigen_df)} seqs) from {args.antigen}")
    else:
        print(f"Using all {len(all_antigen_clones)} clones ({len(antigen_df)} seqs) from {args.antigen}")
    antigen_df["Specificity"] = "antigen"

    # ---- Negative class (second antigen) ----
    negative_df = all_df[all_df["Specificity"] == args.negative].copy()
    print(f"\n=== Assigning clonotypes (MMseqs2, >=80% CDR3 identity): {args.negative} ===")
    negative_df = assign_clone_id_mmseqs2(negative_df)

    # Remove clone_id overlap between the two antigens
    antigen_clone_ids = set(antigen_df["clone_id"])
    before     = len(negative_df)
    negative_df = negative_df[~negative_df["clone_id"].isin(antigen_clone_ids)]
    print(f"Removed {before - len(negative_df)} {args.negative} seqs with clone_id overlap")
    print(f"Using all {negative_df['clone_id'].nunique()} clones ({len(negative_df)} seqs) from {args.negative}")
    negative_df["Specificity"] = "naive"  # keep label for train.py compatibility

    # ---- Combine and split ----
    combined = pd.concat([antigen_df, negative_df], ignore_index=True)
    print(f"\nCombined: {len(combined)} sequences")
    print(combined["Specificity"].value_counts().to_string())

    print(f"\n=== Splitting (cluster-aware, seed={args.seed}) ===")
    train, val, test = split_by_clone(combined, seed=args.seed)

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

    meta = {
        "antigen":            args.antigen,
        "negative":           args.negative,
        "n_clones":           int(antigen_df["clone_id"].nunique()),
        "n_seqs":             len(antigen_df),
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
    parser.add_argument("--antigen",  required=True, choices=["HA", "Qb", "RBD"],
                        help="Positive class antigen")
    parser.add_argument("--negative", required=True, choices=["HA", "Qb", "RBD"],
                        help="Negative class antigen")
    parser.add_argument("--n_clones", type=int, default=None)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out_dir",  required=True)
    main(parser.parse_args())
