#!/bin/bash
#SBATCH -p GPU-3090
#SBATCH --job-name=viz_attention
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
python scripts/attention_viz.py \
    --checkpoint experiments/checkpoints/binary/RBD_vs_naive_all/seed2026/best_model.pt \
    --config     experiments/configs/small.yaml \
    --splits_dir data/splits/clone_filtered_max3 \
    --processed_dir data/processed \
    --n_samples  -1 \
    --out_dir    experiments/logs/attention/seed42/

echo ""
echo "End: $(date)"