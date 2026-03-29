"""
tokenizer.py
------------
Single amino-acid tokenizer for BCR sequences with optional CDR-region tagging.

Vocabulary
----------
20 canonical AAs + special tokens:
  <pad>  – padding                (id 0)
  <unk>  – unknown / non-standard (id 1)
  <cls>  – sequence start         (id 2)
  <eos>  – sequence end           (id 3)
  <mask> – masked token (for MLM) (id 4)
  <cdr1>, <cdr2>, <cdr3>          (id 25-27)  – CDR boundary markers

CDR tagging (optional)
----------------------
If CDR positions are supplied the tokenizer inserts boundary markers:
  <cdr1> ... residues ... <cdr1>
  <cdr2> ... residues ... <cdr2>
  <cdr3> ... residues ... <cdr3>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical, alphabetical

SPECIAL_TOKENS: Dict[str, int] = {
    "<pad>":  0,
    "<unk>":  1,
    "<cls>":  2,
    "<eos>":  3,
    "<mask>": 4,
}

CDR_TOKENS: Dict[str, int] = {
    "<cdr1>": 25,
    "<cdr2>": 26,
    "<cdr3>": 27,
}


def _build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab: Dict[str, int] = {}
    vocab.update(SPECIAL_TOKENS)
    for i, aa in enumerate(AMINO_ACIDS, start=5):   # ids 5-24
        vocab[aa] = i
    vocab.update(CDR_TOKENS)
    id2tok = {v: k for k, v in vocab.items()}
    return vocab, id2tok


# ---------------------------------------------------------------------------
# CDR annotation dataclass
# ---------------------------------------------------------------------------

@dataclass
class CDRAnnotation:
    """
    Zero-based, half-open intervals [start, end) for the three CDR regions
    within the raw (un-tokenized) sequence string.

    Example
    -------
    seq   = "EVQLVES...CDR1...FR2...CDR2...FR3...CDR3...FR4"
    annot = CDRAnnotation(cdr1=(5, 12), cdr2=(25, 33), cdr3=(51, 60))
    """
    cdr1: Tuple[int, int]
    cdr2: Tuple[int, int]
    cdr3: Tuple[int, int]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class BCRTokenizer:
    """
    Single-AA tokenizer for BCR / antibody sequences.

    Parameters
    ----------
    max_length : int
        Maximum total token length (including <cls> / <eos>).
    add_special_tokens : bool
        Prepend <cls> and append <eos>.
    tag_cdrs : bool
        Wrap CDR regions with boundary marker tokens when CDRAnnotation
        is supplied to encode().
    """

    def __init__(
        self,
        max_length: int = 150,
        add_special_tokens: bool = True,
        tag_cdrs: bool = True,
    ) -> None:
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.tag_cdrs = tag_cdrs

        self.vocab, self.id2tok = _build_vocab()
        self.vocab_size = len(self.vocab)

        self.pad_id  = self.vocab["<pad>"]
        self.unk_id  = self.vocab["<unk>"]
        self.cls_id  = self.vocab["<cls>"]
        self.eos_id  = self.vocab["<eos>"]
        self.mask_id = self.vocab["<mask>"]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(
        self,
        sequence: str,
        cdr: Optional[CDRAnnotation] = None,
        padding: bool = False,
        return_tensors: bool = False,
    ) -> Dict[str, List[int] | Tensor]:
        """
        Tokenize a single BCR sequence.

        Returns dict with keys:
            input_ids       – token ids
            attention_mask  – 1 for real tokens, 0 for padding
            cdr_mask        – 1 for CDR positions, 0 elsewhere
        """
        sequence = sequence.upper().strip()
        sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", sequence)

        tokens = self._insert_cdr_tags(sequence, cdr) if (self.tag_cdrs and cdr) else list(sequence)

        if self.add_special_tokens:
            tokens = ["<cls>"] + tokens + ["<eos>"]

        tokens = tokens[: self.max_length]

        input_ids      = [self.vocab.get(t, self.unk_id) for t in tokens]
        attention_mask = [1] * len(input_ids)
        cdr_mask       = self._build_cdr_mask(tokens)

        pad_len = self.max_length - len(input_ids)
        if padding and pad_len > 0:
            input_ids      += [self.pad_id] * pad_len
            attention_mask += [0]           * pad_len
            cdr_mask       += [0]           * pad_len

        result = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "cdr_mask":       cdr_mask,
        }

        if return_tensors:
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}

        return result

    def encode_batch(
        self,
        sequences: List[str],
        cdrs: Optional[List[Optional[CDRAnnotation]]] = None,
        return_tensors: bool = True,
    ) -> Dict[str, List | Tensor]:
        """
        Tokenize a batch of sequences, padding to the longest in the batch.
        """
        if cdrs is None:
            cdrs = [None] * len(sequences)

        assert len(sequences) == len(cdrs), "sequences and cdrs must have equal length"

        encoded = [
            self.encode(seq, cdr=cdr, padding=False)
            for seq, cdr in zip(sequences, cdrs)
        ]

        max_len = min(max(len(e["input_ids"]) for e in encoded), self.max_length)

        batch: Dict[str, List] = {"input_ids": [], "attention_mask": [], "cdr_mask": []}
        for e in encoded:
            pad_len = max_len - len(e["input_ids"])
            batch["input_ids"].append(      e["input_ids"]      + [self.pad_id] * pad_len)
            batch["attention_mask"].append( e["attention_mask"] + [0]           * pad_len)
            batch["cdr_mask"].append(       e["cdr_mask"]       + [0]           * pad_len)

        if return_tensors:
            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        return batch

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token ids back to an amino-acid string."""
        tokens = [self.id2tok.get(i, "<unk>") for i in ids]
        if skip_special_tokens:
            special = set(SPECIAL_TOKENS.keys()) | set(CDR_TOKENS.keys())
            tokens = [t for t in tokens if t not in special]
        return "".join(tokens)

    def format_encode(self, encoded_dict: Dict[str, List[int] | Tensor]) -> str:
        """将 encode 的结果转换为易读的对齐字符串"""
        # 确保是 list 格式
        ids = encoded_dict["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        masks = encoded_dict["cdr_mask"]
        if isinstance(masks, torch.Tensor):
            masks = masks.tolist()
            
        tokens = [self.id2tok.get(i, "<unk>") for i in ids]
        
        # 构造输出
        lines = []
        lines.append(f"{'Index':<6} | {'Token':<8} | {'ID':<4} | {'CDR' if any(masks) else ''}")
        lines.append("-" * 30)
        for i, (t, idx, m) in enumerate(zip(tokens, ids, masks)):
            cdr_mark = " [CDR]" if m == 1 else ""
            lines.append(f"{i:<6} | {t:<8} | {idx:<4} |{cdr_mark}")
        
        return "\n".join(lines)
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_cdr_tags(self, sequence: str, cdr: CDRAnnotation) -> List[str]:
        """Insert CDR boundary markers into the token list."""
        regions = sorted([
            (cdr.cdr1[0], cdr.cdr1[1], "<cdr1>"),
            (cdr.cdr2[0], cdr.cdr2[1], "<cdr2>"),
            (cdr.cdr3[0], cdr.cdr3[1], "<cdr3>"),
        ])

        tokens: List[str] = []
        cursor = 0
        for start, end, tag in regions:
            tokens.extend(list(sequence[cursor:start]))
            tokens.append(tag)
            tokens.extend(list(sequence[start:end]))
            tokens.append(tag)
            cursor = end
        tokens.extend(list(sequence[cursor:]))
        return tokens

    def _build_cdr_mask(self, tokens: List[str]) -> List[int]:
        """1 for residues inside a CDR region, 0 elsewhere."""
        mask = [0] * len(tokens)
        inside = False
        for i, tok in enumerate(tokens):
            if tok in CDR_TOKENS:
                inside = not inside
            elif inside:
                mask[i] = 1
        return mask

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"BCRTokenizer(vocab_size={self.vocab_size}, "
            f"max_length={self.max_length}, "
            f"tag_cdrs={self.tag_cdrs})"
        )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tok = BCRTokenizer(max_length=30, tag_cdrs=True)
    print(tok)

    seq = "EVQLVESGGGLVQPGGSLRLSCAAS"
    cdr = CDRAnnotation(cdr1=(5, 10), cdr2=(14, 18), cdr3=(20, 24))

    out = tok.encode(seq, cdr=cdr, padding=True, return_tensors=True)
    print("input_ids     :", out["input_ids"])
    print("attention_mask:", out["attention_mask"])
    print("cdr_mask      :", out["cdr_mask"])
    print("decoded       :", tok.decode(out["input_ids"].tolist()))

    batch = tok.encode_batch([seq, seq[:15]], cdrs=[cdr, None])
    print("\nbatch input_ids shape:", batch["input_ids"].shape)