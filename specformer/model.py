"""
model.py
--------
SpecFormer: a small Transformer encoder for antigen-specificity
classification of BCR amino-acid sequences.

Architecture
------------
  Embedding layer  (vocab → d_model, + learned positional encoding)
      ↓
  N × Transformer encoder layers  (self-attention + FFN)
      ↓
  [CLS] token representation  (position 0)
      ↓
  Classification head  (Linear → LayerNorm → GELU → Linear)
      ↓
  Logits  (num_classes,)

Default config (small)
----------------------
  vocab_size  : 28   (from BCRTokenizer)
  d_model     : 256
  n_heads     : 4
  n_layers    : 4
  d_ff        : 1024  (4 × d_model)
  dropout     : 0.1
  max_seq_len : 150
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class LearnedPositionalEncoding(nn.Module):
    """Learned (not sinusoidal) positional embeddings — simpler and works well
    for fixed-length biological sequences."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, d_model)
        L = x.size(1)
        positions = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        return x + self.embedding(positions)


# ---------------------------------------------------------------------------
# CDR-aware token embedding
# ---------------------------------------------------------------------------

class BCREmbedding(nn.Module):
    """
    Token embedding + positional encoding + optional CDR type embedding.

    The CDR type embedding adds a learned bias to tokens inside CDR regions,
    giving the model an explicit signal about CDR vs framework positions
    (analogous to token-type embeddings in BERT).
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int,
        max_seq_len: int,
        dropout:     float = 0.1,
        use_cdr_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.token_emb    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc      = LearnedPositionalEncoding(max_seq_len, d_model)
        self.use_cdr_embedding = use_cdr_embedding
        if use_cdr_embedding:
            # 0 = framework, 1 = CDR region
            self.cdr_type_emb = nn.Embedding(2, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: Tensor, cdr_mask: Tensor) -> Tensor:
        # input_ids, cdr_mask: (B, L)
        x = self.token_emb(input_ids)           # (B, L, d_model)
        x = self.pos_enc(x)
        if self.use_cdr_embedding:
            x = x + self.cdr_type_emb(cdr_mask)
        x = self.norm(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Two-layer MLP on top of the [CLS] token."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, cls_repr: Tensor) -> Tensor:
        # cls_repr: (B, d_model)  →  logits: (B, num_classes)
        return self.net(cls_repr)


# ---------------------------------------------------------------------------
# SpecFormer
# ---------------------------------------------------------------------------

class SpecFormer(nn.Module):
    """
    Parameters
    ----------
    num_classes  : number of antigen-specificity classes
    vocab_size   : BCRTokenizer vocabulary size (default 28)
    d_model      : embedding / hidden dimension
    n_heads      : number of attention heads (d_model must be divisible)
    n_layers     : number of Transformer encoder layers
    d_ff         : feed-forward inner dimension (default 4 × d_model)
    dropout      : dropout rate applied throughout
    max_seq_len  : maximum sequence length (must match tokenizer)
    use_cdr_embedding : add CDR-type embedding (recommended)
    """

    def __init__(
        self,
        num_classes:       int,
        vocab_size:        int   = 28,
        d_model:           int   = 256,
        n_heads:           int   = 4,
        n_layers:          int   = 4,
        d_ff:              int   = 1024,
        dropout:           float = 0.1,
        max_seq_len:       int   = 150,
        use_cdr_embedding: bool  = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.embedding = BCREmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_cdr_embedding=use_cdr_embedding,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,       # (B, L, d_model) convention
            norm_first=True,        # pre-LayerNorm: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.head = ClassificationHead(d_model, num_classes, dropout)

        self._init_weights()

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:      Tensor,
        attention_mask: Tensor,
        cdr_mask:       Tensor,
        return_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """
        Parameters
        ----------
        input_ids        : (B, L)  token ids
        attention_mask   : (B, L)  1=real token, 0=padding
        cdr_mask         : (B, L)  1=CDR region, 0=framework
        return_attentions: bool    if True, also return list of attention weights

        Returns
        -------
        logits : (B, num_classes)
        attentions (optional) : list of (B, n_heads, L, L) per layer
        """
        # Embedding
        x = self.embedding(input_ids, cdr_mask)      # (B, L, d_model)

        key_padding_mask = (attention_mask == 0).bool()  # (B, L)

        if return_attentions:
            # Manual layer-by-layer forward to capture attention weights
            attentions = []
            for layer in self.encoder.layers:
                # self_attn returns (attn_output, attn_weights)
                src = layer.norm1(x) if layer.norm_first else x
                attn_out, attn_w = layer.self_attn(
                    src, src, src,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,  # (B, n_heads, L, L)
                )
                attentions.append(attn_w.detach().cpu())
                # Complete the layer forward manually
                x = x + layer.dropout1(attn_out)
                if not layer.norm_first:
                    x = layer.norm1(x)
                # FFN
                src2 = layer.norm2(x) if layer.norm_first else x
                src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src2))))
                x = x + layer.dropout2(src2)
                if not layer.norm_first:
                    x = layer.norm2(x)

            cls_repr = x[:, 0, :]
            logits   = self.head(cls_repr)
            return logits, attentions
        else:
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
            cls_repr = x[:, 0, :]
            return self.head(cls_repr)

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform init for linear layers, zeros for biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"SpecFormer("
            f"d_model={self.embedding.token_emb.embedding_dim}, "
            f"n_layers={len(self.encoder.layers)}, "
            f"n_heads={self.encoder.layers[0].self_attn.num_heads}, "
            f"params={self.num_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from specformer.tokenizer import BCRTokenizer, CDRAnnotation

    tok   = BCRTokenizer(max_length=150, tag_cdrs=True)
    model = SpecFormer(num_classes=3, vocab_size=len(tok))
    print(model)
    print(f"Trainable parameters: {model.num_parameters():,}")

    # Fake batch
    seq  = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR"
    cdr  = CDRAnnotation(cdr1=(25, 33), cdr2=(50, 58), cdr3=(95, 110))
    batch = tok.encode_batch([seq] * 4, cdrs=[cdr] * 4)

    logits = model(
        input_ids      = batch["input_ids"],
        attention_mask = batch["attention_mask"],
        cdr_mask       = batch["cdr_mask"],
    )
    print(f"Input shape:  {batch['input_ids'].shape}")
    print(f"Output shape: {logits.shape}")   # should be (4, 3)
    print(f"Logits: {logits}")