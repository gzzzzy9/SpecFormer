#!/bin/bash
# learning_curve.sh
# Run binary classification experiments for all antigens and n_clones
# to generate learning curve data.
#
# Usage: bash scripts/learning_curve.sh

PROCESSED_DIR="data/processed"
CONFIG="experiments/configs/small.yaml"
N_CLONES_LIST="100 200 500 1000 2000 5000 all"
ANTIGENS="HA Qb RBD"

for ANTIGEN in $ANTIGENS; do
    for N in $N_CLONES_LIST; do

        if [ "$N" = "all" ]; then
            N_ARG=""
            EXP_NAME="binary/${ANTIGEN}_vs_naive_all"
        else
            N_ARG="--n_clones $N"
            EXP_NAME="binary/${ANTIGEN}_vs_naive_${N}"
        fi

        OUT_DIR="data/splits/${EXP_NAME}"

        echo "=== Preprocessing: ${ANTIGEN} vs naive, n_clones=${N} ==="
        python scripts/preprocess_binary.py \
            --processed_dir $PROCESSED_DIR \
            --antigen $ANTIGEN \
            $N_ARG \
            --out_dir $OUT_DIR

        echo "=== Training: ${ANTIGEN} vs naive, n_clones=${N} ==="
        sbatch --export=ALL,\
SEED=42,\
CONFIG=$CONFIG,\
SPLITS_DIR=$OUT_DIR \
            --job-name=sf_${ANTIGEN}_${N} \
            --output=experiments/logs/binary/${ANTIGEN}_${N}_%j.out \
            scripts/run_slurm.sh

    done
done

echo "All jobs submitted."
