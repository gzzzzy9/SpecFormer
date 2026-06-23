#!/bin/bash
#SBATCH -p titan
#SBATCH --job-name=extract_plm_embeddings_esm2
#SBATCH --output=experiments/logs/plm_baseline/esm2/slurm_out/%x_%j_out.txt
#SBATCH --error=experiments/logs/plm_baseline/esm2/slurm_out/%x_%j_err.txt
#SBATCH -n 10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1


# ---- Environment ----
source ~/.bashrc
conda activate specformer

python scripts/extract_plm_embeddings.py \
    --model_path  model/esm2_t30_150M_UR50D \
    --model_type  esm2 \
    --splits_dir  data/splits/binary/RBD_vs_naive_all \
    --out_dir     experiments/logs/plm_baseline/esm2/ \
    --batch_size  32