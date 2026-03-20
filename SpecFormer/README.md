# SpecFormer

A Transformer-based protein language model for learning antigen-specificity patterns in BCR (B cell receptor) amino acid sequences.

## Overview

SpecFormer trains on antigen-specific BCR sequences to discover sequence motifs and patterns associated with antigen binding specificity.

## Project Structure

```
SpecFormer/
├── data/
│   ├── raw/          # Raw AIRR-format BCR sequences
│   ├── processed/    # Tokenized, tensor-ready data
│   └── splits/       # Train / val / test splits
├── specformer/
│   ├── model.py      # Transformer encoder + classification head
│   ├── dataset.py    # BCR sequence dataset & dataloader
│   ├── trainer.py    # Training loop, validation, early stopping
│   ├── tokenizer.py  # Amino acid tokenizer + CDR-aware masking
│   ├── metrics.py    # AUC, accuracy, specificity metrics
│   └── utils.py      # Seeding, logging, checkpoint utilities
├── scripts/
│   ├── preprocess.py # Raw data → processed tensors
│   ├── train.py      # Training entry point
│   └── evaluate.py   # Inference + attention visualization
├── experiments/
│   ├── configs/      # YAML hyperparameter configs
│   ├── checkpoints/  # Saved model weights
│   └── logs/         # Wandb / tensorboard logs
├── notebooks/        # EDA and visualization notebooks
├── requirements.txt
├── environment.yml
└── setup.py
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate specformer

# Preprocess data
python scripts/preprocess.py --input data/raw/ --output data/processed/

# Train
python scripts/train.py --config experiments/configs/base.yaml

# Evaluate
python scripts/evaluate.py --checkpoint experiments/checkpoints/best_model.pt
```
