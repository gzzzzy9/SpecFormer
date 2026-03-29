#!/bin/bash
#SBATCH -p GPU-3090
#SBATCH --job-name=eval_model
#SBATCH --output=experiments/logs/%x_%j_out.txt
#SBATCH --error=experiments/logs/%x_%j_err.txt
#SBATCH -n 10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-3090


# ---- Environment ----
source ~/.bashrc
conda activate DL

# ---- Working directory ----
cd $SLURM_SUBMIT_DIR

# ---- Log job info ----
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ---- Run ----
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/small_cdr_tag/seed42/best_model.pt \
    --config experiments/configs/small.yaml \
    --out_dir    experiments/logs/eval/ \
    --splits_dir data/splits/clone_filtered

python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/small_no_cdr_tag/seed42/best_model.pt \
    --config experiments/configs/small_no_cdr_tag.yaml \
    --out_dir    experiments/logs/eval/ \
    --splits_dir data/splits/clone_filtered


# python scripts/evaluate.py \
#     --checkpoint experiments/checkpoints/small/seed123/best_model.pt \
#     --config experiments/configs/small.yaml \
#     --out_dir    experiments/logs/eval/ \
#     --splits_dir data/splits/clone_filtered_max3

# python scripts/evaluate.py \
#     --checkpoint experiments/checkpoints/small/seed2025/best_model.pt \
#     --config experiments/configs/small.yaml \
#     --out_dir    experiments/logs/eval/ \
#     --splits_dir data/splits/clone_filtered_max3

echo ""
echo "End: $(date)"