"""
plot_learning_curve.py
----------------------
Collect metrics from binary experiments across multiple seeds
and plot learning curves with mean ± SD.

Usage
-----
python scripts/plot_learning_curve.py \
    --eval_dir experiments/logs/binary/ \
    --splits_dir data/splits/binary/ \
    --out      experiments/logs/learning_curve.png
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
# 获取字体文件的绝对路径
font_path = os.path.abspath('fonts/helvetica-255/Helvetica.ttf')

# 检查文件是否存在
if os.path.exists(font_path):
    # 核心：直接把这个路径注册到 Matplotlib
    fm.fontManager.addfont(font_path)
    # 设置为默认字体
    plt.rcParams['font.sans-serif'] = ['Helvetica']
else:
    print(f"警告：找不到字体文件 {font_path}，将使用系统默认字体")

ANTIGENS      = ["HA", "Qb", "RBD"]
N_CLONES_LIST = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, "all"]
SEEDS         = [42, 123, 2026]
COLORS        = {"HA": "#54d5c7", "Qb": "#edba38", "RBD": "#f65150"}
MARKERS       = {"HA": "^", "Qb": "s", "RBD": "o"}
METRICS       = ["auroc", "f1", "precision", "recall"]
METRIC_LABELS = {
    "auroc":     "AUROC",
    "f1":        "F1",
    "precision": "Precision",
    "recall":    "Recall",
}


def collect_results(eval_dir: str, splits_dir: str) -> dict:
    """
    Returns:
        {antigen: {actual_n: {metric: [val_seed1, val_seed2, ...]}}}
    """
    results = {ag: {} for ag in ANTIGENS}

    for antigen in ANTIGENS:
        for n in N_CLONES_LIST:
            # Get actual n_clones from meta.json (same for all seeds)
            meta_path = Path(splits_dir) / f"{antigen}_vs_naive_{n}" / "meta.json"
            if not meta_path.exists():
                print(f"  [SKIP meta] {meta_path}")
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            actual_n = meta.get("n_clones", n)

            metric_vals = {m: [] for m in METRICS}
            n_found = 0

            for seed in SEEDS:
                path = (Path(eval_dir) / f"{antigen}_vs_naive_{n}"
                        / f"seed{seed}" / "results.json")
                if not path.exists():
                    print(f"  [SKIP] {path}")
                    continue
                with open(path) as f:
                    res = json.load(f)

                antigen_cls = res["per_class"].get("antigen", {})
                metric_vals["auroc"].append(res.get("antigen_auroc", float("nan")))
                metric_vals["f1"].append(antigen_cls.get("f1", float("nan")))
                metric_vals["precision"].append(antigen_cls.get("precision", float("nan")))
                metric_vals["recall"].append(antigen_cls.get("recall", float("nan")))
                n_found += 1

            if n_found > 0:
                results[antigen][actual_n] = metric_vals
                auroc_mean = np.mean(metric_vals["auroc"])
                auroc_std  = np.std(metric_vals["auroc"])
                print(f"  {antigen} n={actual_n}: "
                      f"AUROC={auroc_mean:.4f} ± {auroc_std:.4f}  "
                      f"F1={np.mean(metric_vals['f1']):.4f} ± {np.std(metric_vals['f1']):.4f}  "
                      f"Precision={np.mean(metric_vals['precision']):.4f} ± {np.std(metric_vals['precision']):.4f}  "
                      f"Recall={np.mean(metric_vals['recall']):.4f} ± {np.std(metric_vals['recall']):.4f} "
                      f"(n_seeds={n_found})")

    return results


import matplotlib.pyplot as plt
import numpy as np

def plot(results: dict, out_path: str) -> None:
    # 为了让 2x2 的子图接近正方形，figsize 设为宽高接近的比例
    # 增加高度以留出底部图例的空间
    fig, axes = plt.subplots(2, 2, figsize=(11, 15))
    axes_flat = axes.flatten()

    for ax_idx, metric in enumerate(METRICS):
        ax = axes_flat[ax_idx]

        for antigen in ANTIGENS:
            data = results[antigen]
            if not data:
                continue

            xs    = sorted(data.keys())
            means = [np.nanmean(data[x][metric]) for x in xs]
            stds  = [np.nanstd(data[x][metric])  for x in xs]

            ax.plot(xs, means,
                    marker=MARKERS[antigen],
                    color=COLORS[antigen],
                    label=antigen,
                    linewidth=2,
                    markersize=7)

            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=COLORS[antigen],
                            alpha=0.15)

            # 标注最后一个点
            ax.annotate(f"{means[-1]:.3f}",
                        xy=(xs[-1], means[-1]),
                        xytext=(6, 0),
                        textcoords="offset points",
                        fontsize=9,
                        color=COLORS[antigen])

        # 核心设置
        ax.set_xscale("log")
        ax.set_title(METRIC_LABELS[metric], fontsize=16, fontweight="bold", pad=0)
        
        # 根据你的要求设置 y 轴范围
        ax.set_ylim(0.5 if ax_idx == 0 else 0, 1.05)
        
        # 移除子图各自的 xlabel，稍后统一设置
        ax.set_xlabel("")
        
        # 设置网格
        ax.grid(True, alpha=0.2, linestyle="--")

        # 移除右上边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 强制子图为正方形 (Adjustable 参数允许在保持比例的同时填充 layout)
        ax.set_box_aspect(1) 

    # --- 统一全局设置 ---

    # 1. 统一 X 轴标签 (y 坐标根据布局微调)
    fig.supxlabel("Number of antigen clones (Train+Val+Test)", fontsize=18, y=0.08)
    fig.suptitle(f"Learning curves of antigen-specificity detection\n"
        f"(mean ± SD, {len(SEEDS)} seeds)",
        fontsize=18, y=1.01
    )

    # 2. 提取图例并统一放置
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='lower center', 
               ncol=len(ANTIGENS), 
               bbox_to_anchor=(0.5, 0.02), 
               fontsize=18, 
               frameon=False)

    # 3. 调整子图间距，rect 为整体内容留出顶部和底部的空白
    # hspace 增加垂直间距防止标题和坐标轴重叠
    plt.tight_layout(rect=[0, 0.1, 1, 0.98], h_pad=3, w_pad=2)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",   default="experiments/logs/binary/")
    parser.add_argument("--splits_dir", default="data/splits/binary/")
    parser.add_argument("--out",        default="experiments/logs/learning_curve.png")
    args = parser.parse_args()

    print("=== Collecting results ===")
    results = collect_results(args.eval_dir, args.splits_dir)
    plot(results, args.out) 