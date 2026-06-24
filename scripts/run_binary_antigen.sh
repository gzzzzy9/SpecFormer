#!/bin/bash
#SBATCH -p 4090
#SBATCH --job-name=binary_classification_RBD_vs_Qb
#SBATCH --output=experiments/logs/binary/slurms_out/%x_%j_out.txt
#SBATCH --error=experiments/logs/binary/slurms_out/%x_%j_err.txt
#SBATCH -n 20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate specformer
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
export PYTHONUNBUFFERED=1
# ── Config ───────────────────────────────────────────────────────────────────
ANTIGEN1=${ANTIGEN1:-Qb}
ANTIGEN2=${ANTIGEN2:-RBD}
SEED=${SEED:-42}
CONFIG=${CONFIG:-experiments/configs/small.yaml}

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Antigen1:  $ANTIGEN1"
echo "Antigen2:  $ANTIGEN2"
echo "Seed:      $SEED"
echo ""

EXP_NAME="${ANTIGEN1}_vs_${ANTIGEN2}_all"
SPLITS_DIR="data/splits/binary/${EXP_NAME}_seed${SEED}"
CKPT_DIR="experiments/checkpoints/binary/${EXP_NAME}/seed${SEED}"
EVAL_DIR="experiments/logs/binary/${EXP_NAME}/seed${SEED}"

mkdir -p experiments/logs/binary
mkdir -p $CKPT_DIR
mkdir -p $EVAL_DIR

# ── Step 1: Preprocess ───────────────────────────────────────────────────────
echo "=== Step 1: Preprocessing ==="
python -u scripts/preprocess_binary_antigen.py \
    --antigen1   $ANTIGEN1 \
    --antigen2  $ANTIGEN2 \
    --seed      $SEED \
    --out_dir   $SPLITS_DIR

if [ $? -ne 0 ]; then
    echo "Preprocessing failed. Exiting."
    exit 1
fi
echo ""

# ── Step 2: Train ────────────────────────────────────────────────────────────
echo "=== Step 2: Training ==="
python -u scripts/train.py \
    --config     $CONFIG \
    --seed       $SEED \
    --splits_dir $SPLITS_DIR \
    --save_dir   $CKPT_DIR

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi
echo ""

# ── Step 3: Evaluate ─────────────────────────────────────────────────────────
echo "=== Step 3: Evaluating ==="
python -u scripts/evaluate.py \
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