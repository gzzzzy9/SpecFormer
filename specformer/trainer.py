"""
trainer.py
----------
Training loop for SpecFormer with:
  - class-weighted CrossEntropyLoss (handles label imbalance)
  - linear warmup + cosine LR schedule
  - early stopping on val loss
  - automatic class weights from label_map.json + split CSV counts
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Class weight helper
# ---------------------------------------------------------------------------

def compute_class_weights(
    train_csv: str | Path,
    label_map: Dict[str, int],
    device: torch.device,
) -> Tensor:
    """
    Compute inverse-frequency class weights from the training split.

    Formula:  w_c = (1 / count_c) / sum(1 / count_c')
    Weights are normalised so they sum to 1.

    Parameters
    ----------
    train_csv  : path to data/splits/train.csv
    label_map  : {"RBD": 0, "Qb": 1, "HA": 2, ...}
    device     : torch device

    Returns
    -------
    Tensor of shape (num_classes,) on `device`
    """
    df = pd.read_csv(train_csv)
    counts = df["Specificity"].value_counts()

    num_classes = len(label_map)
    weights = torch.zeros(num_classes)
    for label, idx in label_map.items():
        count = counts.get(label, 1)          # fallback to 1 to avoid div/0
        weights[idx] = 1.0 / count

    weights = weights / weights.sum()         # normalise
    print("Class weights:")
    for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
        print(f"  {label:>8} (class {idx}): {weights[idx]:.6f}  "
              f"[n={counts.get(label, 0)}]")
    return weights.to(device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Parameters
    ----------
    model          : nn.Module (SpecFormer)
    train_loader   : DataLoader for training split
    val_loader     : DataLoader for validation split
    label_map      : {"RBD": 0, ...}
    train_csv      : path to train.csv (for computing class weights)
    lr             : peak learning rate
    weight_decay   : AdamW weight decay
    warmup_steps   : linear warmup steps before cosine decay
    epochs         : max training epochs
    patience       : early stopping patience (epochs with no val improvement)
    save_dir       : directory to save best checkpoint
    device         : torch device; auto-detected if None
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        label_map:    Dict[str, int],
        train_csv:    str | Path,
        lr:           float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int   = 1000,
        epochs:       int   = 100,
        patience:     int   = 10,
        save_dir:     str   = "experiments/checkpoints/",
        device:       Optional[torch.device] = None,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.label_map    = label_map
        self.epochs       = epochs
        self.patience     = patience
        self.save_dir     = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        # Multi-GPU: wrap with DataParallel if multiple GPUs available
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
            self.model = nn.DataParallel(self.model)
        else:
            print(f"Using device: {self.device}")

        # ---- Loss with class weights ----
        class_weights = compute_class_weights(train_csv, label_map, self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ---- Optimiser ----
        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # ---- LR schedule: linear warmup → cosine decay ----
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, epochs * len(train_loader) - warmup_steps),
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # ---- Tracking ----
        self.best_val_loss  = float("inf")
        self.patience_count = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, list]:
        """Run the full training loop. Returns history dict."""
        print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
              f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}")
        print("-" * 68)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   train=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            lr  = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
                  f"{val_loss:>8.4f}  {val_acc:>6.2%}  {lr:>8.2e}  {elapsed:>5.1f}s")

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss  = val_loss
                self.patience_count = 0
                self._save_checkpoint(epoch, val_loss, val_acc)
            else:
                self.patience_count += 1
                if self.patience_count >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs).")
                    break

        print(f"\nBest val loss: {self.best_val_loss:.4f}")
        return self.history

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        correct    = 0
        total      = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                cdr_mask       = batch["cdr_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cdr_mask=cdr_mask,
                )                                   # (B, num_classes)

                loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                total_loss += loss.item() * labels.size(0)
                preds       = logits.argmax(dim=-1)
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float) -> None:
        path = self.save_dir / "best_model.pt"
        torch.save(
            {
                "epoch":      epoch,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "label_map":  self.label_map,
            },
            path,
        )
        print(f"  ✓ Saved best model  (epoch {epoch}, val_loss={val_loss:.4f})")


# ---------------------------------------------------------------------------
# Convenience: load weights from saved checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> Dict:
    """Load a saved checkpoint into model. Returns the checkpoint dict."""
    device = device or torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.2%})")
    return ckpt