#!/bin/bash
#SBATCH -p titan
#SBATCH --job-name=preprocess_binary_antigen_Qb_vs_RBD
#SBATCH --output=experiments/logs/binary/slurms_out/%x_%j_out.txt
#SBATCH --error=experiments/logs/binary/slurms_out/%x_%j_err.txt
#SBATCH -n 25
#SBATCH --mem=32G

# ---- Environment ----
source ~/.bashrc
conda activate specformer

export PYTHONUNBUFFERED=1

python -u scripts/preprocess_binary_antigen.py \
    --antigen1   Qb \
    --antigen2  RBD \
    --seed      2026 \
    --out_dir   data/splits/binary_v2/Qb_vs_RBD_all_seed2026