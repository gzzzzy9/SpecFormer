"""
preprocess.py
-------------
Merge all per-sample full_aa_seq_info.csv files with the master sample sheet,
filter, deduplicate, assign clone IDs, and save clone-aware train/val/test splits.

Clone-aware splitting
---------------------
Same clone = same V gene + J gene + CDR3 amino acid sequence.
All sequences from the same clone are kept in the same split to prevent
clone leakage between train and test sets.

Usage
-----
python scripts/preprocess.py \
    --master      data/raw/sample_sheet.tsv \
    --nanopore_dir /path/to/Nanopore_results/ \
    --out_dir     data/processed/ \
    --splits_dir  data/splits/

Directory structure expected
----------------------------
Nanopore_results/
    <Batch>/
        <Library>/
            <Barcode>/
                full_aa_seq_info.csv

Master sheet columns (tab-separated):
    Batch  Library  Barcode  Ear No.  Specificity  Tissue  Cells  Genotype  Sample
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_SEQ_COLS = [
    "aaSeq", "aaSeqCDR3", "bestVHit", "bestJHit",
    "uniqueMoleculeCount",
    "CDR1Begin", "CDR1End",
    "CDR2Begin", "CDR2End",
    "CDR3Begin", "CDR3End",
]

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
SEED        = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_master(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep="\t", encoding="gbk")
    df.columns = df.columns.str.strip()
    df["Specificity"] = df["Specificity"].astype(str).str.strip()
    return df
def load_sample_seqs(nanopore_dir: str, batch: str, library: str, barcode: str) -> pd.DataFrame | None:
    path = Path(nanopore_dir) / str(batch) / str(library) / str(barcode) / "full_aa_seq_info.csv"
    if not path.exists():
        print(f"  [WARN] not found: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_SEQ_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] {path} missing columns: {missing}")
        return None
    return df[REQUIRED_SEQ_COLS]


def merge_naive(master: pd.DataFrame, nanopore_dir: str) -> pd.DataFrame:
    """
    Extract sequences from naive B cell samples.
    Naive B rows are identified by Sample column containing 'naive B'
    and empty Specificity.
    """
    naive_rows = master[
        master["Sample"].astype(str).str.lower().str.contains("naive", na=False) &
        (master["Specificity"].isna() | (master["Specificity"].astype(str).str.strip() == "") |
         (master["Specificity"].astype(str).str.strip() == "nan"))
    ]
    print(f"Naive B samples in master: {len(naive_rows)}")

    frames = []
    for _, row in naive_rows.iterrows():
        batch   = row["Batch"]
        library = row["Library"]
        barcode = row["Barcode"]

        seqs = load_sample_seqs(nanopore_dir, batch, library, barcode)
        if seqs is None:
            continue
        seqs = seqs.copy()
        seqs["Specificity"] = "naive"
        seqs["Batch"]       = batch
        seqs["Library"]     = library
        seqs["Barcode"]     = barcode
        frames.append(seqs)
        print(f"  Loaded {len(seqs):>6} seqs  [{batch} / {library} / {barcode}]  label=naive")

    if not frames:
        print("[WARN] No naive B sequences found.")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    print(f"Total naive B sequences: {len(merged)}")
    return merged


def merge_spec(master: pd.DataFrame, nanopore_dir: str) -> pd.DataFrame:
    labeled = master[
        master["Specificity"].notna() &
        (master["Specificity"] != "") &
        (master["Specificity"] != "nan")
    ]
    print(f"Labeled samples in master: {len(labeled)}")

    frames = []
    for _, row in labeled.iterrows():
        batch       = row["Batch"]
        library     = row["Library"]
        barcode     = row["Barcode"]
        specificity = row["Specificity"]

        seqs = load_sample_seqs(nanopore_dir, batch, library, barcode)
        if seqs is None:
            continue
        seqs = seqs.copy()
        seqs["Specificity"] = specificity
        seqs["Batch"]       = batch
        seqs["Library"]     = library
        seqs["Barcode"]     = barcode
        frames.append(seqs)
        print(f"  Loaded {len(seqs):>6} seqs  [{batch} / {library} / {barcode}]  label={specificity}")

    if not frames:
        raise RuntimeError("No sequence files found. Check --nanopore_dir path.")

    merged = pd.concat(frames, ignore_index=True)
    print(f"\nTotal before dedup: {len(merged)}")
    return merged


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate full-length sequences, keep first occurrence."""
    df = df.dropna(subset=["aaSeq"])
    df = df.sort_values("aaSeq").drop_duplicates(subset="aaSeq", keep="first")
    print(f"Total after dedup:  {len(df)}")
    return df.reset_index(drop=True)


def assign_clone_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define clone as: V gene (ignoring allele) + J gene (ignoring allele) + CDR3 aa sequence.
    """
    v_gene = df["bestVHit"].str.split("*").str[0]
    j_gene = df["bestJHit"].str.split("*").str[0]
    cdr3   = df["aaSeqCDR3"].fillna("")

    df = df.copy()
    df["clone_id"] = v_gene + "_" + j_gene + "_" + cdr3

    n_clones = df["clone_id"].nunique()
    n_seqs   = len(df)
    print(f"Unique clones: {n_clones}  ({n_seqs} sequences, "
          f"avg {n_seqs/n_clones:.1f} seqs/clone)")
    return df


def remove_conflicting_clones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all sequences belonging to clones that appear in more than one
    Specificity label. These cross-reactive clones send contradictory signals
    to the model.
    """
    clone_labels = df.groupby("clone_id")["Specificity"].nunique()
    conflict_clones = clone_labels[clone_labels > 1].index
    before = len(df)
    n_clones_removed = len(conflict_clones)
    df = df[~df["clone_id"].isin(conflict_clones)]
    print(f"Removed {before - len(df)} sequences from {n_clones_removed} "
          f"conflicting clones (same clone, multiple labels)")
    return df.reset_index(drop=True)


def max_n_per_clone(df: pd.DataFrame, max_n: int = 3) -> pd.DataFrame:
    """
    Keep maximal n abundant sequences per clone
    (highest uniqueMoleculeCount). This removes intra-clone redundancy and ensures
    each clone contributes exactly one training example.
    """
    if max_n == -1:
        print('No filtering!')
        return df
    before = len(df)
    # Sort by uniqueMoleculeCount descending, keep first (highest) per clone
    df = df.sort_values("uniqueMoleculeCount", ascending=False)
    df = df.groupby("clone_id").head(max_n)  
    print('Sequences: {} --> {}'.format(before, len(df)))
    return df.reset_index(drop=True)


def split_by_clone(df: pd.DataFrame):
    """
    Pure-numpy stratified clone-aware split.
    Splits clones (not sequences) into train/val/test so that all sequences
    from the same clone land in exactly one split.
    Stratified by Specificity label at the clone level.
    """
    rng = np.random.default_rng(SEED)

    # Get majority label per clone
    clone_label = (
        df.groupby("clone_id")["Specificity"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"Specificity": "clone_label"})
    )

    train_clones, val_clones, test_clones = [], [], []

    # Stratified split: process each label separately
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

    train = df[df["clone_id"].isin(train_set)]
    val   = df[df["clone_id"].isin(val_set)]
    test  = df[df["clone_id"].isin(test_set)]

    print(f"\nClone-aware split:")
    print(f"  train: {len(train):>6} seqs  ({train['clone_id'].nunique():>5} clones)")
    print(f"  val:   {len(val):>6} seqs  ({val['clone_id'].nunique():>5} clones)")
    print(f"  test:  {len(test):>6} seqs  ({test['clone_id'].nunique():>5} clones)")

    # Verify zero clone overlap
    assert len(train_set & val_set)  == 0, "train/val clone overlap!"
    assert len(train_set & test_set) == 0, "train/test clone overlap!"
    assert len(val_set   & test_set) == 0, "val/test clone overlap!"
    print("  ✓ Zero clone overlap between splits")

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_label_map(df: pd.DataFrame) -> dict[str, int]:
    labels = sorted(df["Specificity"].unique())
    return {lbl: i for i, lbl in enumerate(labels)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    # If exp_name is given, save splits to data/splits/<exp_name>/
    if args.exp_name:
        splits_dir = str(Path(args.splits_dir) / args.exp_name)
        print(f"Experiment name: {args.exp_name}")
        print(f"Splits will be saved to: {splits_dir}")
    else:
        splits_dir = args.splits_dir
    os.makedirs(splits_dir, exist_ok=True)
    args.splits_dir = splits_dir

    print("=== Loading master sheet ===")
    master = load_master(args.master)
    print(f"Total rows in master: {len(master)}")

    print("\n=== Merging antigen-specific sequences ===")
    merged = merge_spec(master, args.nanopore_dir)

    print("\n=== Extracting naive B sequences ===")
    naive_df = merge_naive(master, args.nanopore_dir)
    if len(naive_df) > 0:
        naive_df = deduplicate(naive_df)
        naive_path = Path(args.out_dir) / "naive_sequences.csv"
        naive_df.to_csv(naive_path, index=False)
        print(f"Naive B sequences saved → {naive_path}  ({len(naive_df)} sequences)")

        if args.include_naive:
            print("\n=== Removing naive clones that overlap with antigen-specific clones ===")
            # First assign clone IDs to naive sequences
            naive_df = assign_clone_id(naive_df)
            antigen_clone_ids = set(assign_clone_id(merged)["clone_id"])
            before = len(naive_df)
            naive_df = naive_df[~naive_df["clone_id"].isin(antigen_clone_ids)]
            print(f"Removed {before - len(naive_df)} naive sequences "
                  f"overlapping with antigen-specific clones")
            print(f"Remaining naive sequences: {len(naive_df)}")

            print("\n=== Merging naive B into antigen-specific sequences ===")
            merged = pd.concat([merged, naive_df], ignore_index=True)
            print(f"Combined total: {len(merged)} sequences")

    print("\n=== Deduplicating (full aa sequence) ===")
    deduped = deduplicate(merged)

    print("\n=== Assigning clone IDs ===")
    deduped = assign_clone_id(deduped)

    print("\n=== Removing conflicting clones ===")
    deduped = remove_conflicting_clones(deduped)

    print("\n=== One sequence per clone (keep highest uniqueMoleculeCount) ===")
    deduped = max_n_per_clone(deduped, max_n=int(args.keep_max_n))

    print("\n=== Label distribution (after dedup) ===")
    print(deduped["Specificity"].value_counts().to_string())

    # Label map
    label_map = build_label_map(deduped)
    label_map_path = Path(args.out_dir) / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nLabel map: {label_map}  →  {label_map_path}")

    # Save full processed file
    seq_out = Path(args.out_dir) / "sequences.csv"
    deduped.to_csv(seq_out, index=False)
    print(f"Saved processed sequences → {seq_out}")

    # Clone-aware split
    print("\n=== Clone-aware splitting ===")
    train, val, test = split_by_clone(deduped)

    print("\n=== Label distribution per split ===")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = split["Specificity"].value_counts().to_dict()
        print(f"  {name}: {dist}")

    train.to_csv(Path(args.splits_dir) / "train.csv", index=False)
    val.to_csv(  Path(args.splits_dir) / "val.csv",   index=False)
    test.to_csv( Path(args.splits_dir) / "test.csv",  index=False)
    print(f"\nSplits saved → {args.splits_dir}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master",       required=True, help="Master sample sheet (TSV)")
    parser.add_argument("--nanopore_dir", required=True, help="Root of Nanopore_results/")
    parser.add_argument("--out_dir",      default="data/processed/")
    parser.add_argument("--splits_dir",   default="data/splits/")
    parser.add_argument("--exp_name",     default=None,
                        help="Experiment name, creates data/splits/<exp_name>/. "
                             "If not set, saves directly to splits_dir.")
    parser.add_argument("--include_naive", action="store_true",
                        help="Include naive B as a 4th class (label=naive).")
    parser.add_argument("--keep_max_n", default="3",
                        help="Keep max n sequences per ")
    main(parser.parse_args())