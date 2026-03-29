#!/bin/bash
#SBATCH -p GPU-3090
#SBATCH --job-name=train_specformer
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
SEED=${SEED:-42}
CONFIG=${CONFIG:-experiments/configs/small.yaml}
SPLITS_DIR=${SPLITS_DIR:-data/splits/clone_filtered_max3}

echo "Seed: $SEED; Model config: $CONFIG; Splits: $SPLITS_DIR"

python scripts/train.py \
    --config     $CONFIG \
    --seed       $SEED \
    --splits_dir $SPLITS_DIR

echo ""
echo "End: $(date)"