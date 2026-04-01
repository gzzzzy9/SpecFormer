"""
dataset.py
----------
PyTorch Dataset and DataLoader factory for SpecFormer.

Reads the split CSVs produced by preprocess.py, tokenizes each sequence
with BCRTokenizer, and returns batches of:
    input_ids       – (B, L)  token ids
    attention_mask  – (B, L)  1=real token, 0=pad
    cdr_mask        – (B, L)  1=inside CDR region
    labels          – (B,)    integer class ids
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from specformer.tokenizer import BCRTokenizer, CDRAnnotation


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCRDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str | Path
        Path to a split CSV (train.csv / val.csv / test.csv).
    label_map : dict[str, int]
        Mapping from specificity string to integer class id.
        Load from data/processed/label_map.json.
    tokenizer : BCRTokenizer
    use_junction : bool
        If True, CDR3 coordinates are treated as the full junction
        (C104 + CDR3 + W118). No coordinate adjustment is needed;
        just document which convention your data uses.
    """

    def __init__(
        self,
        csv_path: str | Path,
        label_map: Dict[str, int],
        tokenizer: BCRTokenizer,
        use_junction: bool = True,
    ) -> None:
        self.tokenizer    = tokenizer
        self.label_map    = label_map
        self.use_junction = use_junction

        df = pd.read_csv(csv_path)
        self._validate_columns(df)

        before = len(df)
        df = df.dropna(subset=["aaSeq"]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"  [WARN] Dropped {dropped} rows with missing aaSeq")

        self.sequences  = df["aaSeq"].astype(str).tolist()
        self.labels     = df["Specificity"].map(label_map).tolist()
        self.cdr_coords = df[
            ["CDR1Begin", "CDR1End", "CDR2Begin", "CDR2End", "CDR3Begin", "CDR3End"]
        ].values.tolist()

        assert None not in self.labels, (
            "Some Specificity values are not in label_map. "
            "Re-run preprocess.py or check label_map.json."
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        seq  = self.sequences[idx]
        coords = self.cdr_coords[idx]

        cdr = CDRAnnotation(
            cdr1=(int(coords[0]), int(coords[1])),
            cdr2=(int(coords[2]), int(coords[3])),
            cdr3=(int(coords[4]), int(coords[5])),
        )

        encoded = self.tokenizer.encode(
            seq,
            cdr=cdr,
            padding=True,
            return_tensors=True,
        )

        encoded["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoded

    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        required = [
            "aaSeq", "Specificity",
            "CDR1Begin", "CDR1End",
            "CDR2Begin", "CDR2End",
            "CDR3Begin", "CDR3End",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    splits_dir:   str | Path,
    processed_dir: str | Path,
    tokenizer:    BCRTokenizer,
    batch_size:   int = 64,
    num_workers:  int = 4,
    use_junction: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    splits_dir    : directory containing train.csv, val.csv, test.csv
    processed_dir : directory containing label_map.json
    tokenizer     : BCRTokenizer instance
    batch_size    : sequences per batch
    num_workers   : DataLoader worker processes
    use_junction  : passed through to BCRDataset

    Returns
    -------
    dict with keys "train", "val", "test"
    """
    splits_dir    = Path(splits_dir)
    processed_dir = Path(processed_dir)

    # Prefer label_map.json from splits_dir (binary experiments),
    # fall back to processed_dir (standard 3-class experiments)
    splits_label_map    = splits_dir    / "label_map.json"
    processed_label_map = processed_dir / "label_map.json"
    label_map_path = splits_label_map if splits_label_map.exists() else processed_label_map
    with open(label_map_path) as f:
        label_map = json.load(f)

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found, skipping.")
            continue

        dataset = BCRDataset(
            csv_path=csv_path,
            label_map=label_map,
            tokenizer=tokenizer,
            use_junction=use_junction,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),   # keep batches uniform during training
        )
        print(f"{split:>5} DataLoader: {len(dataset):>6} sequences, "
              f"{len(loaders[split]):>4} batches (batch_size={batch_size})")

    return loaders


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tok = BCRTokenizer(max_length=150, tag_cdrs=True)

    # Minimal smoke test with synthetic data (no files needed)
    import tempfile, os
    dummy = pd.DataFrame({
        "aaSeq":       ["EVQLVESGGGLVQPGG", "QVTLKESGPGILKPSQ"],
        "Specificity": ["RBD", "Qb"],
        "CDR1Begin":   [25, 25], "CDR1End": [33, 35],
        "CDR2Begin":   [50, 52], "CDR2End": [58, 59],
        "CDR3Begin":   [95, 96], "CDR3End": [110, 117],
    })
    label_map = {"RBD": 0, "Qb": 1}

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "train.csv")
        dummy.to_csv(csv_path, index=False)

        ds = BCRDataset(csv_path, label_map, tok)
        print(f"Dataset length: {len(ds)}")
        sample = ds[0]
        for k, v in sample.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")