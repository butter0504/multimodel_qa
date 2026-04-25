"""
CIFAR-10N/100N 标签错误检测评估脚本
===================================
使用带有真实噪声标签的 CIFAR-10N/100N 数据集评估检测器性能。

评估流程：
    1. 加载数据集（包含噪声标签 given_label 和真实标签 clean_label）
    2. 运行检测器（检测器只看到噪声标签）
    3. 使用评估器对比检测结果与真实标签
    4. 计算精确率、召回率、F1分数、修正准确率等指标

使用方式：
    python test_cifar_n.py --dataset CIFAR-10N --sample_size 1000
"""

import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import Config
from modules.image_detector import ImageDetector
from modules.data_adapter import DataAdapter
from modules.evaluator import DetectorEvaluator, create_detector_outputs_from_result


def evaluate_cifar_n(dataset_name: str, data_path: str, sample_size: int = None,
                     threshold: float = None):
    """
    评估 CIFAR-10N/100N 数据集上的标签错误检测性能。

    Parameters
    ----------
    dataset_name : str
        数据集名称 ('CIFAR-10N' 或 'CIFAR-100N')
    data_path : str
        数据集路径
    sample_size : int
        评估样本数
    threshold : float
        标签质量分数阈值（可选，覆盖配置文件）

    Returns
    -------
    Dict[str, Any]
        评估结果
    """
    print("=" * 70)
    print(f"评估数据集: {dataset_name}")
    print("=" * 70)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据集...")
    adapter = DataAdapter()
    data = adapter.load(
        source_type='standard_dataset',
        dataset_name=dataset_name,
        data_path=data_path,
        sample_size=sample_size
    )

    images = data['images']
    noisy_labels = data['labels']
    clean_labels = data['clean_labels']
    label_names = data['label_names']
    metadata = data['metadata']

    print(f"  样本数: {len(images)}")
    print(f"  类别数: {len(label_names)}")
    print(f"  实际噪声率: {metadata['noise_rate']:.2%}")
    print(f"  实际错误数: {metadata['actual_errors']}")

    # Step 2: 配置检测器
    print("\n[Step 2] 配置检测器...")
    cfg = Config()
    if threshold is not None:
        cfg.set('detection.threshold', threshold)
        print(f"  使用自定义阈值: {threshold}")
    else:
        print(f"  使用默认阈值: {cfg.get('detection.threshold')}")

    # Step 3: 运行检测器
    print("\n[Step 3] 运行标签错误检测...")
    detector = ImageDetector(cfg)

    start_time = time.time()
    result = detector.detect(
        images,
        labels=noisy_labels,
        modules=['label_error']
    )
    elapsed = time.time() - start_time

    print(f"  检测耗时: {elapsed:.2f} 秒")

    le = result.get('label_error', {})
    if 'error' in le:
        print(f"  检测失败: {le['error']}")
        return None

    print(f"  检测到的可疑标签数: {le.get('error_count', 0)}")
    print(f"  检测到的错误率: {le.get('error_rate', 0):.2%}")

    # Step 4: 创建评估输入
    print("\n[Step 4] 准备评估数据...")
    detector_outputs = create_detector_outputs_from_result(
        result, noisy_labels, list(range(len(images)))
    )

    clean_label_map = {idx: label for idx, label in enumerate(clean_labels)}

    # Step 5: 运行评估
    print("\n[Step 5] 运行评估...")
    evaluator = DetectorEvaluator()
    eval_results = evaluator.evaluate(detector_outputs, clean_label_map)

    # Step 6: 打印报告
    print("\n" + "=" * 70)
    evaluator.print_report()

    # Step 7: 详细分析
    print("\n[详细分析]")
    print("-" * 50)

    error_indices = le.get('error_indices', [])
    suggested_labels = le.get('suggested_labels', {})

    correct_detections = 0
    wrong_detections = 0
    correct_fixes = 0
    wrong_fixes = 0

    for idx in error_indices:
        noisy = noisy_labels[idx]
        clean = clean_labels[idx]
        suggested = suggested_labels.get(str(idx))

        if noisy != clean:
            correct_detections += 1
        else:
            wrong_detections += 1

        if suggested is not None:
            if suggested == clean:
                correct_fixes += 1
            else:
                wrong_fixes += 1

    print(f"检测器标记的可疑样本: {len(error_indices)}")
    print(f"  - 正确检测（确实是错误）: {correct_detections}")
    print(f"  - 误报（实际正确但被标记）: {wrong_detections}")
    print(f"  - 检测精确率: {correct_detections/len(error_indices):.2%}" if error_indices else "  - 检测精确率: N/A")

    print(f"\n标签修正情况:")
    print(f"  - 正确修正: {correct_fixes}")
    print(f"  - 错误修正: {wrong_fixes}")
    print(f"  - 修正准确率: {correct_fixes/(correct_fixes+wrong_fixes):.2%}" if (correct_fixes+wrong_fixes) > 0 else "  - 修正准确率: N/A")

    # Step 8: 展示示例
    print("\n[示例展示] Top 10 可疑样本:")
    print("-" * 50)
    scores = le.get('label_quality_scores', [])
    sorted_indices = sorted(error_indices, key=lambda i: scores[i] if i < len(scores) else 1)[:10]

    for idx in sorted_indices:
        noisy = noisy_labels[idx]
        clean = clean_labels[idx]
        suggested = suggested_labels.get(str(idx))
        score = scores[idx] if idx < len(scores) else 0

        noisy_name = label_names[noisy] if noisy < len(label_names) else str(noisy)
        clean_name = label_names[clean] if clean < len(label_names) else str(clean)
        suggested_name = label_names[suggested] if suggested is not None and suggested < len(label_names) else str(suggested)

        is_correct_detection = "V" if noisy != clean else "X"
        is_correct_fix = "V" if suggested == clean else "X" if suggested is not None else "-"

        print(f"  #{idx:4d} | 分数: {score:.3f} | "
              f"噪声: {noisy_name:12s} -> 真实: {clean_name:12s} | "
              f"建议: {suggested_name:12s} | 检测:{is_correct_detection} 修正:{is_correct_fix}")

    return {
        'dataset': dataset_name,
        'sample_size': len(images),
        'actual_noise_rate': metadata['noise_rate'],
        'detected_error_count': le.get('error_count', 0),
        'detected_error_rate': le.get('error_rate', 0),
        'evaluation': eval_results,
        'correct_detections': correct_detections,
        'wrong_detections': wrong_detections,
        'correct_fixes': correct_fixes,
        'wrong_fixes': wrong_fixes,
        'elapsed_time': elapsed,
    }


def run_threshold_analysis(dataset_name: str, data_path: str, sample_size: int = 1000):
    """
    运行不同阈值下的性能分析。

    Parameters
    ----------
    dataset_name : str
        数据集名称
    data_path : str
        数据集路径
    sample_size : int
        样本数
    """
    print("\n" + "=" * 70)
    print("阈值敏感性分析")
    print("=" * 70)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = []

    for threshold in thresholds:
        print(f"\n测试阈值: {threshold}")
        result = evaluate_cifar_n(dataset_name, data_path, sample_size, threshold)
        if result:
            results.append({
                'threshold': threshold,
                'precision': result['evaluation'].get('precision', 0),
                'recall': result['evaluation'].get('recall', 0),
                'f1': result['evaluation'].get('f1', 0),
                'fix_accuracy': result['evaluation'].get('fix_accuracy', 0),
            })

    print("\n" + "=" * 70)
    print("阈值分析结果汇总")
    print("=" * 70)
    print(f"\n{'阈值':<10} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'修正准确率':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['threshold']:<10.2f} {r['precision']:<12.3f} {r['recall']:<12.3f} "
              f"{r['f1']:<12.3f} {r['fix_accuracy']:<12.3f}")

    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n最佳阈值 (F1最高): {best_f1['threshold']:.2f}, F1={best_f1['f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10N/100N 标签错误检测评估')
    parser.add_argument('--dataset', type=str, default='CIFAR-10N',
                       choices=['CIFAR-10N', 'CIFAR-100N'],
                       help='数据集名称')
    parser.add_argument('--data_path', type=str,
                       default=r'c:\Users\abc14\Desktop\毕业设计\代码\multimodel_qa\data\raw',
                       help='数据集路径')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='评估样本数 (0=全部)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='标签质量分数阈值')
    parser.add_argument('--threshold_analysis', action='store_true',
                       help='运行阈值敏感性分析')

    args = parser.parse_args()

    sample_size = None if args.sample_size == 0 else args.sample_size

    if args.threshold_analysis:
        run_threshold_analysis(args.dataset, args.data_path, sample_size or 1000)
    else:
        evaluate_cifar_n(args.dataset, args.data_path, sample_size, args.threshold)


if __name__ == '__main__':
    main()
