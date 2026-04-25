"""
检测器性能评估脚本
==================
独立于检测器运行，用于评估标签错误检测器的性能。

评估流程：
    1. 加载检测器输出结果（包含 given_label, suggested_label, has_issue 等）
    2. 加载真实标签（clean_label，来自 CIFAR-10N 等数据集）
    3. 对比检测结果与真实情况
    4. 计算精确率、召回率、F1分数、修正准确率等指标

使用方式：
    from modules.evaluator import DetectorEvaluator

    evaluator = DetectorEvaluator()
    results = evaluator.evaluate(detector_outputs, clean_label_map)
    evaluator.print_report()
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class DetectorEvaluator:
    """
    检测器性能评估器
    ================
    评估标签错误检测器的性能，包括问题检测能力和标签修正能力。

    评估指标：
        - 精确率（Precision）：检测器标记为有问题的样本中，真正有问题的比例
        - 召回率（Recall）：真正有问题的样本中，检测器找出的比例
        - F1分数：精确率和召回率的调和平均
        - 修正准确率（Fix Accuracy）：检测器给出建议标签时，建议正确的比例
        - 错误发现率（Detection Rate）：实际错误中，检测器发现了多少

    参考：
        - Northcutt, C. G., et al. (2021). Confident Learning: Estimating
          Uncertainty in Dataset Labels. JAIR, 70, 1373-1411.
    """

    def __init__(self):
        self.results = None
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
        self.correct_fixes = 0
        self.total_detected_issues = 0
        self.total_actual_issues = 0
        self.total_samples = 0

    def evaluate(self, detector_outputs: List[Dict[str, Any]],
                 clean_label_map: Dict[int, Any],
                 label_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        评估检测器性能。

        Parameters
        ----------
        detector_outputs : List[Dict[str, Any]]
            检测器输出列表，每个元素包含：
            - image_id: 图像索引（整数）
            - given_label: 原始标签（检测器看到的）
            - has_issue: 布尔值，True 表示检测器认为标签有问题
            - confidence: 置信度（可选，用于阈值分析）
            - suggested_label: 建议标签（如果有问题）
        clean_label_map : Dict[int, Any]
            真实标签映射表 {image_index: clean_label}
        label_names : Optional[List[str]]
            标签名称列表，用于混淆矩阵显示

        Returns
        -------
        Dict[str, float]
            评估指标字典
        """
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
        self.correct_fixes = 0
        self.total_detected_issues = 0
        self.total_actual_issues = 0
        self.total_samples = len(detector_outputs)

        fix_details = []

        for output in detector_outputs:
            idx = output.get('image_id', output.get('index', 0))
            given = output.get('given_label', output.get('original_label'))
            clean = clean_label_map.get(idx)
            has_issue = output.get('has_issue', output.get('quality_score', 1.0) < 0.6)
            confidence = output.get('confidence', output.get('quality_score', 0.5))
            suggested = output.get('suggested_label')

            if clean is None:
                continue

            is_truly_correct = (given == clean)
            self.y_true.append(0 if is_truly_correct else 1)
            self.y_pred.append(1 if has_issue else 0)
            self.y_scores.append(confidence if has_issue else 1 - confidence)

            if has_issue:
                self.total_detected_issues += 1
                if suggested is not None and suggested == clean:
                    self.correct_fixes += 1
                    fix_details.append({
                        'image_id': idx,
                        'given': given,
                        'suggested': suggested,
                        'clean': clean,
                        'correct': True
                    })
                elif suggested is not None:
                    fix_details.append({
                        'image_id': idx,
                        'given': given,
                        'suggested': suggested,
                        'clean': clean,
                        'correct': False
                    })

            if not is_truly_correct:
                self.total_actual_issues += 1

        if len(self.y_true) == 0:
            return {'error': 'No valid samples for evaluation'}

        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)

        fix_accuracy = (
            self.correct_fixes / self.total_detected_issues
            if self.total_detected_issues > 0 else 0
        )

        detection_rate = recall

        try:
            cm = confusion_matrix(self.y_true, self.y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        except Exception:
            tn, fp, fn, tp = 0, 0, 0, 0

        self.results = {
            'total_samples': self.total_samples,
            'actual_issues': self.total_actual_issues,
            'actual_issue_rate': self.total_actual_issues / self.total_samples if self.total_samples > 0 else 0,
            'detected_issues': self.total_detected_issues,
            'detected_issue_rate': self.total_detected_issues / self.total_samples if self.total_samples > 0 else 0,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fix_accuracy': float(fix_accuracy),
            'detection_rate': float(detection_rate),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'correct_fixes': self.correct_fixes,
            'fix_details': fix_details,
        }

        return self.results

    def print_report(self):
        """打印评估报告"""
        if self.results is None:
            print("尚未执行评估，请先调用 evaluate() 方法")
            return

        print("=" * 60)
        print("  检测器性能评估报告")
        print("=" * 60)
        print(f"  总样本数: {self.results['total_samples']}")
        print(f"  实际错误样本数: {self.results['actual_issues']} "
              f"({self.results['actual_issue_rate']:.1%})")
        print(f"  检测器标记的问题数: {self.results['detected_issues']} "
              f"({self.results['detected_issue_rate']:.1%})")
        print("-" * 60)
        print("  问题检测能力:")
        print(f"    准确率 (Accuracy): {self.results['accuracy']:.3f}")
        print(f"    精确率 (Precision): {self.results['precision']:.3f}")
        print(f"    召回率 (Recall): {self.results['recall']:.3f}")
        print(f"    F1分数: {self.results['f1']:.3f}")
        print("-" * 60)
        print("  标签修正能力:")
        print(f"    修正准确率: {self.results['fix_accuracy']:.3f}")
        print(f"    正确修正数: {self.results['correct_fixes']}")
        print("-" * 60)
        print("  混淆矩阵:")
        print(f"    真阴性 (TN): {self.results['true_negatives']} - 正确判定为无问题")
        print(f"    假阳性 (FP): {self.results['false_positives']} - 误报（实际正确但被判有问题）")
        print(f"    假阴性 (FN): {self.results['false_negatives']} - 漏报（实际有问题但未检测出）")
        print(f"    真阳性 (TP): {self.results['true_positives']} - 正确检测出问题")
        print("=" * 60)

        if self.results['precision'] < 0.5:
            print("  [WARN] 精确率偏低：检测器误报过多，建议提高置信阈值")
        if self.results['recall'] < 0.5:
            print("  [WARN] 召回率偏低：检测器漏报过多，建议降低置信阈值")
        if self.results['fix_accuracy'] < 0.5:
            print("  [WARN] 修正准确率偏低：建议标签不准确，分类器能力不足")

    def get_summary_dict(self) -> Dict[str, Any]:
        """获取评估结果摘要字典"""
        if self.results is None:
            return {}
        return {
            'precision': self.results['precision'],
            'recall': self.results['recall'],
            'f1': self.results['f1'],
            'fix_accuracy': self.results['fix_accuracy'],
            'actual_issue_rate': self.results['actual_issue_rate'],
            'detected_issue_rate': self.results['detected_issue_rate'],
        }


def load_cifar10n_labels(data_path: str = './data/CIFAR-10_human.pt') -> Dict[str, Any]:
    """
    加载 CIFAR-10N 数据集的标签。

    Parameters
    ----------
    data_path : str
        CIFAR-10N 标签文件路径

    Returns
    -------
    Dict[str, Any]
        包含 given_labels, clean_labels, label_names 的字典
    """
    import torch
    import os

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CIFAR-10N 标签文件不存在: {data_path}")

    noise_file = torch.load(data_path)

    given_labels = noise_file.get('worse_label', noise_file.get('given_label', None))
    clean_labels = noise_file.get('clean_label', None)

    if given_labels is None or clean_labels is None:
        raise ValueError("标签文件格式不正确，缺少 'worse_label' 或 'clean_label' 字段")

    label_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    clean_label_map = {idx: int(label) for idx, label in enumerate(clean_labels)}
    given_label_map = {idx: int(label) for idx, label in enumerate(given_labels)}

    return {
        'given_labels': given_labels.tolist() if hasattr(given_labels, 'tolist') else list(given_labels),
        'clean_labels': clean_labels.tolist() if hasattr(clean_labels, 'tolist') else list(clean_labels),
        'clean_label_map': clean_label_map,
        'given_label_map': given_label_map,
        'label_names': label_names,
        'num_samples': len(clean_labels),
    }


def create_detector_outputs_from_result(detection_result: Dict[str, Any],
                                        given_labels: List[Any],
                                        image_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    从检测结果创建评估器所需的输出格式。

    Parameters
    ----------
    detection_result : Dict[str, Any]
        ImageDetector.detect() 的返回结果
    given_labels : List[Any]
        原始标签列表
    image_ids : Optional[List[int]]
        图像索引列表，默认为 range(len(given_labels))

    Returns
    -------
    List[Dict[str, Any]]
        评估器所需的 detector_outputs 格式
    """
    label_error_result = detection_result.get('label_error', {})
    error_indices = label_error_result.get('error_indices', [])
    suggested_labels = label_error_result.get('suggested_labels', {})
    quality_scores = label_error_result.get('label_quality_scores', [])

    if image_ids is None:
        image_ids = list(range(len(given_labels)))

    outputs = []
    for i, (img_id, given) in enumerate(zip(image_ids, given_labels)):
        has_issue = i in error_indices
        confidence = quality_scores[i] if i < len(quality_scores) else 0.5
        suggested = suggested_labels.get(str(i)) if has_issue else None

        outputs.append({
            'image_id': img_id,
            'given_label': given,
            'has_issue': has_issue,
            'confidence': confidence,
            'suggested_label': suggested,
        })

    return outputs
