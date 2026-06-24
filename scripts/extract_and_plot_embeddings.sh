#!/bin/bash
#SBATCH --job-name=extract_embeddings_and_plot_embeddings
#SBATCH -n 1
#SBATCH -p 3090
#SBATCH --gres=gpu:1
#SBATCH --output=../experiments/logs/binary/slurms_out/extract_embeddings/%x_%j_out.txt
#SBATCH --error=../experiments/logs/binary/slurms_out/extract_embeddings/%x_%j_err.txt

exec {BASH_XTRACEFD}>&1
set -x
work_dir=".."

# 第一步：提取 embedding
python scripts/extract_embeddings.py \
    --checkpoint  experiments/checkpoints/binary/Qb_vs_RBD_all/seed42/best_model.pt \
    --config      experiments/configs/small.yaml \
    --splits_dir  data/splits/binary/Qb_vs_RBD_all_seed42 \
    --out_dir     experiments/logs/embeddings/Qb_vs_RBD_all/seed42

# 第二步：画 UMAP
python plot_umap.py \
    --emb_dir ${work_dir}/experiments/logs/binary/embeddings/Qb_vs_naive/ \
    --out     ${work_dir}/experiments/logs/binary/embeddings/Qb_vs_naive/umap.png