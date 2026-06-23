#!/bin/bash
#SBATCH -p titan
#SBATCH --job-name=train_classifier_on_embeddings_esm2
#SBATCH --output=experiments/logs/plm_baseline/esm2/slurm_out/%x_%j_out.txt
#SBATCH --error=experiments/logs/plm_baseline/esm2/slurm_out/%x_%j_err.txt
#SBATCH -n 10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1


# ---- Environment ----
source ~/.bashrc
conda activate specformer

python scripts/train_classifier_on_embeddings.py \
    --emb_dir    experiments/logs/plm_baseline/esm2/ \
    --out_dir    experiments/logs/plm_baseline/esm2/ \
    --model_name ESM2_150M