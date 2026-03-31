#!/bin/bash
#SBATCH --job-name=binary_classification
#SBATCH --output=experiments/logs/binary/%x_%j_out.txt
#SBATCH --error=experiments/logs/binary/%x_%j_err.txt
#SBATCH -n 10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-3090


# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate DL
cd $SLURM_SUBMIT_DIR

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Antigen:   $ANTIGEN"
echo "N_clones:  $N_CLONES"
echo ""

# ── Config ───────────────────────────────────────────────────────────────────
ANTIGEN=${ANTIGEN:-RBD}
N_CLONES=${N_CLONES:-all}
SEED=${SEED:-42}
CONFIG=${CONFIG:-experiments/configs/small.yaml}
BALANCE=${BALANCE:-""}  # Empty by default, set BALANCE="--balance_naive" when using --balance_naive

if [ "$N_CLONES" = "all" ]; then
    N_ARG=""
else
    N_ARG="--n_clones $N_CLONES"
fi

EXP_NAME="${ANTIGEN}_vs_naive_${N_CLONES}"
OUT_DIR="data/splits/binary/${EXP_NAME}"
SPLITS_DIR="data/splits/binary/${EXP_NAME}"
CKPT_DIR="experiments/checkpoints/binary/${EXP_NAME}/seed${SEED}"
EVAL_DIR="experiments/logs/binary/${EXP_NAME}/seed${SEED}"

mkdir -p experiments/logs/binary
mkdir -p $CKPT_DIR
mkdir -p $EVAL_DIR

# ── Step 1: Preprocess ───────────────────────────────────────────────────────
echo "=== Step 1: Preprocessing ==="
python scripts/preprocess_binary.py \
    --antigen $ANTIGEN \
    $N_ARG \
    $BALANCE \
    --out_dir $OUT_DIR

if [ $? -ne 0 ]; then
    echo "Preprocessing failed. Exiting."
    exit 1
fi
echo ""

# ── Step 2: Train ────────────────────────────────────────────────────────────
echo "=== Step 2: Training ==="
python scripts/train.py \
    --config     $CONFIG \
    --seed       $SEED \
    --splits_dir $SPLITS_DIR

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

# Move checkpoint to experiment-specific directory
cp experiments/checkpoints/small/seed${SEED}/best_model.pt $CKPT_DIR/best_model.pt
echo ""

# ── Step 3: Evaluate ─────────────────────────────────────────────────────────
echo "=== Step 3: Evaluating ==="
python scripts/evaluate.py \
    --checkpoint $CKPT_DIR/best_model.pt \
    --config     $CONFIG \
    --splits_dir $SPLITS_DIR \
    --out_dir    $EVAL_DIR

if [ $? -ne 0 ]; then
    echo "Evaluation failed. Exiting."
    exit 1
fi

echo ""
echo "=== Done: ${EXP_NAME} seed=${SEED} ==="
echo "End: $(date)"
