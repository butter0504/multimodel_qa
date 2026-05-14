"""
标签错误检测算法对比实验 — 可视化脚本
====================================
读取 reports/comparison_results.txt 中的实验数据，
生成论文级对比图表，输出到 reports/ 目录。

生成图表：
    1. comparison_table.png  — 对比结果表格
    2. comparison_bar.png    — 分组柱状图（P/R/F1/AUC）
    3. comparison_radar.png  — 雷达图

使用方式：
    python plot_comparison.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pip_packages'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(SCRIPT_DIR, 'reports')

METHODS = ['Random', 'Confidence', 'Loss-based', 'Ours (CL)']
METRICS = ['Precision', 'Recall', 'F1', 'AUC']

DATA = {
    'Random':     {'Precision': 0.401, 'Recall': 0.500, 'F1': 0.445, 'AUC': 0.500, 'Time': 0.0},
    'Confidence': {'Precision': 0.425, 'Recall': 0.209, 'F1': 0.281, 'AUC': 0.532, 'Time': 20.4},
    'Loss-based': {'Precision': 0.665, 'Recall': 0.429, 'F1': 0.522, 'AUC': 0.736, 'Time': 20.5},
    'Ours (CL)':  {'Precision': 0.599, 'Recall': 0.627, 'F1': 0.613, 'AUC': 0.736, 'Time': 40.7},
}

COLORS = ['#8ecae6', '#219ebc', '#fb8500', '#d62828']
HIGHLIGHT_COLOR = '#d62828'


def plot_table():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    col_labels = ['方法', 'Precision', 'Recall', 'F1', 'AUC', '时间(秒)']
    cell_text = []
    cell_colors = []

    for i, method in enumerate(METHODS):
        d = DATA[method]
        row = [
            method,
            f"{d['Precision']:.3f}",
            f"{d['Recall']:.3f}",
            f"{d['F1']:.3f}",
            f"{d['AUC']:.3f}",
            f"{d['Time']:.1f}",
        ]
        cell_text.append(row)

        row_colors = []
        for j, metric_key in enumerate(['Precision', 'Recall', 'F1', 'AUC']):
            values = [DATA[m][metric_key] for m in METHODS]
            max_val = max(values)
            if d[metric_key] == max_val:
                row_colors.append('#ffd6d6')
            else:
                row_colors.append('#ffffff')
        row_colors.insert(0, '#f0f0f0')
        row_colors.append('#ffffff')
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=['#333333'] * len(col_labels),
        cellLoc='center',
        loc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_facecolor('#333333')
        if col == 0 and row > 0:
            cell.set_text_props(fontweight='bold')
        if row > 0 and col == 4:
            val = DATA[METHODS[row - 1]]['F1']
            if val == max(DATA[m]['F1'] for m in METHODS):
                cell.set_text_props(fontweight='bold', color=HIGHLIGHT_COLOR)

    ax.set_title('表 5-1  标签错误检测算法对比实验结果（CIFAR-10N 数据集）',
                 fontsize=14, fontweight='bold', pad=20)

    output_path = os.path.join(REPORTS_DIR, 'comparison_table.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"已保存: {output_path}")


def plot_bar():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        values = [DATA[m][metric] for m in METHODS]

        bars = ax.bar(range(len(METHODS)), values, color=COLORS,
                      width=0.6, edgecolor='white', linewidth=1.2)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, fontsize=9, rotation=15)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim(0, max(values) * 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        max_idx = values.index(max(values))
        bars[max_idx].set_edgecolor(HIGHLIGHT_COLOR)
        bars[max_idx].set_linewidth(2.5)

    fig.suptitle('图 5-1  标签错误检测算法性能对比（CIFAR-10N 数据集）',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    output_path = os.path.join(REPORTS_DIR, 'comparison_bar.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"已保存: {output_path}")


def plot_radar():
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, method in enumerate(METHODS):
        values = [DATA[method][m] for m in METRICS]
        values += values[:1]

        linewidth = 2.5 if method == 'Ours (CL)' else 1.2
        alpha = 0.25 if method == 'Ours (CL)' else 0.08

        ax.plot(angles, values, 'o-', linewidth=linewidth,
                color=COLORS[i], label=method, markersize=5)
        ax.fill(angles, values, alpha=alpha, color=COLORS[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), METRICS, fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('图 5-2  标签错误检测算法雷达图（CIFAR-10N 数据集）',
                 fontsize=14, fontweight='bold', pad=25)

    output_path = os.path.join(REPORTS_DIR, 'comparison_radar.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"已保存: {output_path}")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("生成对比实验可视化图表...")
    plot_table()
    plot_bar()
    plot_radar()
    print("\n全部图表生成完成！输出目录: " + REPORTS_DIR)


if __name__ == '__main__':
    main()
