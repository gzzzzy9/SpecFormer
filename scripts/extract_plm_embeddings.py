"""
extract_plm_embeddings.py
--------------------------
Extract frozen, mean-pooled sequence embeddings from a pretrained protein/
antibody language model (AntiBERTa2, ESM2, etc.) for train/val/test splits,
to be used as baseline features for comparison against SpecFormer.

Usage
-----
# AntiBERTa2 (RoFormer-based, space-separated tokenization)
python scripts/extract_plm_embeddings.py \
    --model_path  model/antiberta2 \
    --model_type  antiberta2 \
    --splits_dir  data/splits/binary/RBD_vs_naive_all \
    --out_dir     experiments/logs/plm_baseline/antiberta2/

# ESM2 (character-level tokenizer, no manual spacing needed)
python scripts/extract_plm_embeddings.py \
    --model_path  model/esm2_t12_35M_UR50D \
    --model_type  esm2 \
    --splits_dir  data/splits/binary/RBD_vs_naive_all \
    --out_dir     experiments/logs/plm_baseline/esm2/

Notes
-----
- Sequences are taken from the `aaSeq` column of train.csv/val.csv/test.csv,
  WITHOUT any CDR boundary tags (standard PLM usage, no SpecFormer-style
  CDR tokens since these PLMs were never trained with them).
- Embeddings = mean pooling over all real residue positions in the last
  hidden layer, excluding special tokens (CLS/EOS/SEP/PAD).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SeqDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def load_model_and_tokenizer(model_path: str, model_type: str):
    if model_type == "antiberta2":
        from transformers import RoFormerTokenizer, RoFormerModel
        tokenizer = RoFormerTokenizer.from_pretrained(model_path)
        model = RoFormerModel.from_pretrained(model_path)
    elif model_type == "esm2":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return tokenizer, model


def format_sequence(seq: str, model_type: str) -> str:
    seq = seq.strip().upper()
    if model_type == "antiberta2":
        # AntiBERTa2 / RoFormer tokenizer expects space-separated amino acids
        return " ".join(seq)
    return seq  # ESM2 tokenizer handles raw sequences (character-level)


@torch.no_grad()
def embed_batch(model, tokenizer, seqs, model_type, device, max_length=160):
    formatted = [format_sequence(s, model_type) for s in seqs]
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # (B, L, D)

    attn_mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)

    # Zero out special tokens (CLS/SEP/EOS) from pooling by excluding first
    # and last valid position per sequence (works for BERT/RoFormer-style
    # [CLS] ... [SEP] and ESM-style <cls> ... <eos> single-special-token wrap)
    seq_lens = inputs["attention_mask"].sum(dim=1)  # (B,)
    pool_mask = attn_mask.clone()
    for i, L in enumerate(seq_lens.tolist()):
        L = int(L)
        if L >= 2:
            pool_mask[i, 0, 0] = 0.0       # exclude CLS/BOS
            pool_mask[i, L - 1, 0] = 0.0   # exclude SEP/EOS

    masked_hidden = hidden * pool_mask
    summed = masked_hidden.sum(dim=1)               # (B, D)
    counts = pool_mask.sum(dim=1).clamp(min=1.0)     # (B, 1)
    mean_pooled = summed / counts

    return mean_pooled.cpu().numpy()


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.model_type} from {args.model_path} ...")
    tokenizer, model = load_model_and_tokenizer(args.model_path, args.model_type)
    model.to(device)
    model.eval()

    for split in ["train", "val", "test"]:
        csv_path = Path(args.splits_dir) / f"{split}.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        seqs = df["aaSeq"].tolist()
        labels = df["Specificity"].tolist()

        print(f"\nExtracting {split}: {len(seqs)} sequences...")
        all_embeddings = []
        batch_size = args.batch_size

        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i + batch_size]
            emb = embed_batch(model, tokenizer, batch_seqs, args.model_type,
                              device, max_length=args.max_length)
            all_embeddings.append(emb)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  {i + len(batch_seqs)}/{len(seqs)} done")

        embeddings = np.concatenate(all_embeddings, axis=0)
        np.save(out_dir / f"{split}_embeddings.npy", embeddings)

        meta = pd.DataFrame({"label": labels})
        meta.to_csv(out_dir / f"{split}_labels.csv", index=False)

        print(f"  Saved {split}: embeddings shape {embeddings.shape}")

    print(f"\nDone. All outputs saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Local path to pretrained PLM checkpoint")
    parser.add_argument("--model_type", required=True,
                        choices=["antiberta2", "esm2"])
    parser.add_argument("--splits_dir", required=True,
                        help="Dir containing train.csv/val.csv/test.csv with aaSeq + Specificity columns")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=160)
    main(parser.parse_args())
