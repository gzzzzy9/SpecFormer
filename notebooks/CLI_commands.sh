# Preprocessing
# Without naive 
python scripts/preprocess.py \
    --master ../Nanopore_results/wetwork_info/Nanopore_wetwork_info.xlsx \
    --nanopore_dir ../Nanopore_results/ \
    --exp_name clone_filtered_max3 \
    > experiments/logs/preprocessing_out_20260327.txt
# With naive
python scripts/preprocess.py \
    --master ../Nanopore_results/wetwork_info/Nanopore_wetwork_info.xlsx \
    --nanopore_dir ../Nanopore_results/ \
    --exp_name clone_filtered_max3 \
    --include_naive \
    > experiments/logs/preprocessing_out_20260327.txt

# 训练
python scripts/train.py \
    --config experiments/configs/small.yaml \
    --splits_dir data/splits/clone_filtered_max3

# 评估（用同一个 splits_dir）
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/small/seed42/best_model.pt \
    --config     experiments/configs/small.yaml \
    --splits_dir data/splits/clone_filtered_max3 \
    --out_dir    experiments/logs/eval/clone_filtered/

python3 -c "
import pandas as pd
for s in ['train','val','test']:
    df = pd.read_csv(f'data/splits/clone_filtered_max3_with_naive/{s}.csv')
    print(s, df['Specificity'].value_counts().to_dict())
"

for ANTIGEN in RBD Qb; do
    for N in 10000 20000; do
        for SEED in 123 2026; do
            sbatch \
                --job-name=${ANTIGEN}_${N} \
                --output=experiments/logs/binary/${ANTIGEN}_${N}_s${SEED}_%j_out.txt \
                --export=ALL,ANTIGEN=$ANTIGEN,N_CLONES=$N,SEED=$SEED \
                scripts/run_binary_classification.sh
        done
    done
done

for ANTIGEN in HA; do
    for N in 100 200 500 1000 2000 all; do
        sbatch \
            --job-name=${ANTIGEN}_${N} \
            --output=experiments/logs/binary/${ANTIGEN}_${N}_%j_out.txt \
            --error=experiments/logs/binary/${ANTIGEN}_${N}_%j_err.txt \
            --export=ALL,ANTIGEN=$ANTIGEN,N_CLONES=$N,SEED=42 \
            scripts/run_binary_classification.sh
    done
done

# 画AUROC, Precision, Recall, F1
python scripts/plot_learning_curve.py \
    --eval_dir   experiments/logs/binary/ \
    --splits_dir data/splits/binary/ \
    --out        experiments/logs/binary/learning_curve.png