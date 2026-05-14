"""
文本检测器验证脚本
==================
使用 AG News 数据集验证 TextDetector 的全部 7 个检测模块是否正常运行。

数据集格式：label, gt, text, group
    - label: 给定标签（含噪声）
    - gt: 真实标签（ground truth）
    - text: 新闻文本
    - group: 分组编号

检测模块：
    1. text_length       - 文本长度检测
    2. duplicate         - 文本重复检测
    3. label_error       - 标签错误检测（需要 labels）
    4. character_anomaly - 字符异常检测
    5. language          - 文本语言检测
    6. perplexity        - 文本困惑度检测
    7. sentiment_consistency - 情感一致性检测（需要 labels）

使用方式：
    python test_text_detector.py
    python test_text_detector.py --sample_size 2000
"""

import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from modules.config import Config
from modules.text_detector import TextDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(SCRIPT_DIR, 'reports')

AGNEWS_LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']


def load_agnews(csv_path, sample_size=None):
    df = pd.read_csv(csv_path)

    if sample_size and sample_size < len(df):
        df = df.head(sample_size)

    texts = df['text'].tolist()
    noisy_labels = df['label'].tolist()
    clean_labels = df['gt'].tolist()

    n_errors = sum(1 for n, c in zip(noisy_labels, clean_labels) if n != c)
    noise_rate = n_errors / len(noisy_labels) if noisy_labels else 0

    print(f"  数据集: AG News ({len(texts)}样本)")
    print(f"  实际标签错误数: {n_errors}")
    print(f"  实际噪声率: {noise_rate:.2%}")

    return texts, noisy_labels, clean_labels


def run_text_detector(texts, labels, modules):
    cfg = Config()
    detector = TextDetector(cfg)

    print(f"\n  运行检测模块: {modules}")
    t0 = time.time()
    result = detector.detect(texts, labels=labels, modules=modules)
    elapsed = time.time() - t0

    print(f"  检测总耗时: {elapsed:.1f}s")
    return result, elapsed


def evaluate_label_error(label_result, noisy_labels, clean_labels):
    y_true = np.array([1 if n != c else 0 for n, c in zip(noisy_labels, clean_labels)])

    if 'error' in label_result:
        print(f"  标签错误检测失败: {label_result['error']}")
        return None

    error_indices = set(label_result['error_indices'])
    y_pred = np.zeros(len(noisy_labels), dtype=int)
    for idx in error_indices:
        if idx < len(y_pred):
            y_pred[idx] = 1

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    quality_scores = label_result.get('label_quality_scores', [])
    if quality_scores:
        scores_arr = np.array(quality_scores)
        correct_scores = scores_arr[y_true == 0]
        error_scores = scores_arr[y_true == 1]
        print(f"  正确标签平均质量分数: {np.mean(correct_scores):.4f}")
        print(f"  错误标签平均质量分数: {np.mean(error_scores):.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detected_count': len(error_indices),
        'actual_error_count': int(np.sum(y_true)),
    }


def print_section(title):
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description='文本检测器验证脚本')
    parser.add_argument('--csv_path', type=str,
                        default=os.path.join(SCRIPT_DIR, 'data', 'uploads',
                                             'samples10000valMed——AGNews.csv'),
                        help='AG News CSV 文件路径')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='采样数量')

    args = parser.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    print_section("文本检测器验证实验")
    print(f"数据集: AG News")

    print("\n[Step 1] 加载数据集...")
    texts, noisy_labels, clean_labels = load_agnews(args.csv_path, args.sample_size)

    all_results = {}

    # ---- 模块 1: 文本长度检测 ----
    print_section("模块 1: 文本长度检测")
    t0 = time.time()
    result_1, _ = run_text_detector(texts, noisy_labels, ['text_length'])
    t1 = time.time() - t0
    metrics_1 = result_1['text_length']['metrics']
    stats = result_1['text_length']['length_distribution']['statistics']
    print(f"  总文本数: {metrics_1['total_texts']}")
    print(f"  过短文本数: {metrics_1['short_text_count']} ({metrics_1['short_text_rate']:.2f}%)")
    print(f"  过长文本数: {metrics_1['long_text_count']} ({metrics_1['long_text_rate']:.2f}%)")
    print(f"  长度异常文本数: {metrics_1['abnormal_length_count']}")
    print(f"  长度统计: min={stats['min']:.0f} max={stats['max']:.0f} "
          f"mean={stats['mean']:.0f} median={stats['median']:.0f}")
    print(f"  耗时: {t1:.1f}s")
    all_results['text_length'] = {
        'status': 'PASS',
        'short_text_count': metrics_1['short_text_count'],
        'long_text_count': metrics_1['long_text_count'],
        'abnormal_length_count': metrics_1['abnormal_length_count'],
        'mean_length': round(stats['mean'], 1),
        'time': round(t1, 1),
    }

    # ---- 模块 2: 文本重复检测 ----
    print_section("模块 2: 文本重复检测")
    t0 = time.time()
    result_2, _ = run_text_detector(texts, noisy_labels, ['duplicate'])
    t2 = time.time() - t0
    metrics_2 = result_2['duplicate']['metrics']
    print(f"  完全重复数: {metrics_2['exact_duplicate_count']} ({metrics_2['exact_duplicate_rate']:.2f}%)")
    print(f"  近似重复数: {metrics_2['approximate_duplicate_count']} ({metrics_2['approximate_duplicate_rate']:.2f}%)")
    print(f"  唯一文本数: {metrics_2['unique_text_count']}")
    print(f"  耗时: {t2:.1f}s")
    all_results['duplicate'] = {
        'status': 'PASS',
        'exact_duplicate_count': metrics_2['exact_duplicate_count'],
        'approximate_duplicate_count': metrics_2['approximate_duplicate_count'],
        'unique_text_count': metrics_2['unique_text_count'],
        'time': round(t2, 1),
    }

    # ---- 模块 3: 标签错误检测 ----
    print_section("模块 3: 标签错误检测（置信学习）")
    t0 = time.time()
    result_3, _ = run_text_detector(texts, noisy_labels, ['label_error'])
    t3 = time.time() - t0
    label_result = result_3['label_error']
    eval_3 = evaluate_label_error(label_result, noisy_labels, clean_labels)
    if eval_3:
        print(f"  检测到标签错误数: {eval_3['detected_count']}")
        print(f"  实际标签错误数: {eval_3['actual_error_count']}")
        print(f"  精确率: {eval_3['precision']:.3f}")
        print(f"  召回率: {eval_3['recall']:.3f}")
        print(f"  F1分数: {eval_3['f1']:.3f}")
    debug = label_result.get('debug_info', {})
    if debug:
        print(f"  特征维度: {debug.get('feature_dim', 'N/A')}")
        print(f"  交叉验证折数: {debug.get('cv_folds_used', 'N/A')}")
        print(f"  平均质量分数: {debug.get('mean_quality_score', 'N/A')}")
    print(f"  耗时: {t3:.1f}s")
    all_results['label_error'] = {
        'status': 'PASS' if eval_3 else 'FAIL',
        'precision': round(eval_3['precision'], 3) if eval_3 else 0,
        'recall': round(eval_3['recall'], 3) if eval_3 else 0,
        'f1': round(eval_3['f1'], 3) if eval_3 else 0,
        'detected_count': eval_3['detected_count'] if eval_3 else 0,
        'time': round(t3, 1),
    }

    # ---- 模块 4: 字符异常检测 ----
    print_section("模块 4: 字符异常检测")
    t0 = time.time()
    result_4, _ = run_text_detector(texts, noisy_labels, ['character_anomaly'])
    t4 = time.time() - t0
    metrics_4 = result_4['character_anomaly']['metrics']
    print(f"  乱码文本数: {metrics_4['garbled_text_count']}")
    print(f"  高特殊字符文本数: {metrics_4['high_special_char_count']}")
    print(f"  首尾空白文本数: {metrics_4['whitespace_padding_count']}")
    print(f"  乱码率: {metrics_4['garbled_text_rate']:.2f}%")
    print(f"  耗时: {t4:.1f}s")
    all_results['character_anomaly'] = {
        'status': 'PASS',
        'garbled_text_count': metrics_4['garbled_text_count'],
        'high_special_char_count': metrics_4['high_special_char_count'],
        'whitespace_padding_count': metrics_4['whitespace_padding_count'],
        'time': round(t4, 1),
    }

    # ---- 模块 5: 文本语言检测 ----
    print_section("模块 5: 文本语言检测")
    t0 = time.time()
    result_5, _ = run_text_detector(texts, noisy_labels, ['language'])
    t5 = time.time() - t0
    metrics_5 = result_5['language']['metrics']
    print(f"  语言分布: {metrics_5['language_distribution']}")
    print(f"  混合语言文本数: {metrics_5['mixed_language_count']}")
    print(f"  非目标语言文本数: {metrics_5['non_target_language_count']}")
    print(f"  混合语言率: {metrics_5['mixed_language_rate']:.2f}%")
    print(f"  耗时: {t5:.1f}s")
    all_results['language'] = {
        'status': 'PASS',
        'language_distribution': metrics_5['language_distribution'],
        'mixed_language_count': metrics_5['mixed_language_count'],
        'time': round(t5, 1),
    }

    # ---- 模块 6: 文本困惑度检测 ----
    print_section("模块 6: 文本困惑度检测")
    t0 = time.time()
    result_6, _ = run_text_detector(texts, noisy_labels, ['perplexity'])
    t6 = time.time() - t0
    metrics_6 = result_6['perplexity']['metrics']
    method = result_6['perplexity'].get('method', 'unknown')
    print(f"  计算方法: {method}")
    print(f"  高困惑度文本数: {metrics_6['high_perplexity_count']}")
    print(f"  平均困惑度: {metrics_6['mean_perplexity']:.2f}")
    print(f"  困惑度阈值: {metrics_6['perplexity_threshold']:.2f}")
    print(f"  耗时: {t6:.1f}s")
    all_results['perplexity'] = {
        'status': 'PASS',
        'method': method,
        'high_perplexity_count': metrics_6['high_perplexity_count'],
        'mean_perplexity': metrics_6['mean_perplexity'],
        'time': round(t6, 1),
    }

    # ---- 模块 7: 情感一致性检测 ----
    print_section("模块 7: 情感一致性检测")
    t0 = time.time()
    result_7, _ = run_text_detector(texts, noisy_labels, ['sentiment_consistency'])
    t7 = time.time() - t0
    metrics_7 = result_7['sentiment_consistency']['metrics']
    print(f"  不一致文本数: {metrics_7['inconsistency_count']}")
    print(f"  不一致率: {metrics_7['inconsistency_rate']:.2f}%")
    print(f"  耗时: {t7:.1f}s")
    all_results['sentiment_consistency'] = {
        'status': 'PASS',
        'inconsistency_count': metrics_7['inconsistency_count'],
        'inconsistency_rate': round(metrics_7['inconsistency_rate'], 2),
        'time': round(t7, 1),
    }

    # ---- 汇总 ----
    print_section("文本检测器验证结果汇总")
    print(f"{'模块':<25} {'状态':<8} {'关键指标':<40} {'耗时(秒)':<10}")
    print("-" * 85)

    module_summary = [
        ('text_length', '文本长度检测',
         f"过短:{all_results['text_length']['short_text_count']} "
         f"过长:{all_results['text_length']['long_text_count']} "
         f"异常:{all_results['text_length']['abnormal_length_count']} "
         f"均值:{all_results['text_length']['mean_length']}"),
        ('duplicate', '文本重复检测',
         f"完全重复:{all_results['duplicate']['exact_duplicate_count']} "
         f"近似重复:{all_results['duplicate']['approximate_duplicate_count']}"),
        ('label_error', '标签错误检测',
         f"P:{all_results['label_error']['precision']} "
         f"R:{all_results['label_error']['recall']} "
         f"F1:{all_results['label_error']['f1']}"),
        ('character_anomaly', '字符异常检测',
         f"乱码:{all_results['character_anomaly']['garbled_text_count']} "
         f"高特殊字符:{all_results['character_anomaly']['high_special_char_count']}"),
        ('language', '文本语言检测',
         f"混合语言:{all_results['language']['mixed_language_count']}"),
        ('perplexity', '文本困惑度检测',
         f"高困惑度:{all_results['perplexity']['high_perplexity_count']} "
         f"均值:{all_results['perplexity']['mean_perplexity']}"),
        ('sentiment_consistency', '情感一致性检测',
         f"不一致:{all_results['sentiment_consistency']['inconsistency_count']} "
         f"率:{all_results['sentiment_consistency']['inconsistency_rate']}%"),
    ]

    for key, name, indicator in module_summary:
        status = all_results[key]['status']
        t = all_results[key]['time']
        print(f"{name:<22} {status:<8} {indicator:<40} {t:<10.1f}")

    # ---- 保存结果 ----
    output_path = os.path.join(REPORTS_DIR, 'text_detector_results.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(" 文本检测器验证实验结果\n")
        f.write(f" 数据集: AG News ({len(texts)}样本)\n")
        f.write("=" * 50 + "\n\n")

        for key, name, indicator in module_summary:
            status = all_results[key]['status']
            t = all_results[key]['time']
            f.write(f"{name}: {status} | {indicator} | {t:.1f}s\n")

        if eval_3:
            f.write(f"\n标签错误检测详细结果:\n")
            f.write(f"  精确率: {eval_3['precision']:.3f}\n")
            f.write(f"  召回率: {eval_3['recall']:.3f}\n")
            f.write(f"  F1分数: {eval_3['f1']:.3f}\n")
            f.write(f"  检测到错误数: {eval_3['detected_count']}\n")
            f.write(f"  实际错误数: {eval_3['actual_error_count']}\n")

    print(f"\n结果已保存到: {output_path}")

    json_path = os.path.join(REPORTS_DIR, 'text_detector_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"JSON 结果已保存到: {json_path}")

    return all_results


if __name__ == '__main__':
    main()
