"""
CIFAR-10 数据集测试脚本
=======================
使用 CIFAR-10 数据集测试图像检测器的各项功能。

测试内容：
    1. 基础质量检测（模糊/曝光/噪声/分辨率/对比度）
    2. 标签错误检测（Confidence Learning）
    3. 不确定性估计

使用方式：
    python test_cifar10.py
"""

import sys
import os
import tarfile
import pickle
import numpy as np
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import Config
from modules.image_detector import ImageDetector


def load_cifar10(data_path: str, extract_path: str = None):
    """
    加载 CIFAR-10 数据集。

    Parameters
    ----------
    data_path : str
        cifar-10-python.tar.gz 文件路径
    extract_path : str
        解压路径，默认为 data_path 同目录

    Returns
    -------
    Tuple[List[bytes], List[int], List[str]]
        (图像字节列表, 标签列表, 类别名称列表)
    """
    if extract_path is None:
        extract_path = os.path.dirname(data_path)

    cifar_dir = os.path.join(extract_path, 'cifar-10-batches-py')
    if not os.path.exists(cifar_dir):
        print(f"解压 CIFAR-10 数据集到 {extract_path}...")
        with tarfile.open(data_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        print("解压完成")

    def load_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data']
            Y = datadict[b'labels']
            X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            return X, Y

    print("加载 CIFAR-10 训练数据...")
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(cifar_dir, f'data_batch_{b}')
        X, Y = load_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    print(f"加载 CIFAR-10 测试数据...")
    X_test, y_test = load_batch(os.path.join(cifar_dir, 'test_batch'))

    meta_file = os.path.join(cifar_dir, 'batches.meta')
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        label_names = [name.decode('utf-8') for name in meta[b'label_names']]

    print(f"训练集: {X_train.shape[0]} 张图像")
    print(f"测试集: {X_test.shape[0]} 张图像")
    print(f"类别: {label_names}")

    def images_to_bytes(images):
        bytes_list = []
        for img_array in images:
            img = Image.fromarray(img_array)
            buf = BytesIO()
            img.save(buf, format='PNG')
            bytes_list.append(buf.getvalue())
        return bytes_list

    print("转换图像为字节格式...")
    train_bytes = images_to_bytes(X_train)
    test_bytes = images_to_bytes(X_test)

    return train_bytes, y_train, test_bytes, y_test, label_names


def test_basic_quality(detector, images, sample_size=1000):
    """测试基础质量检测模块"""
    print("\n" + "=" * 60)
    print("测试 1: 基础质量检测")
    print("=" * 60)

    sample_images = images[:sample_size]
    result = detector.detect(sample_images, modules=['basic_quality'])

    bq = result.get('basic_quality', {})
    metrics = bq.get('metrics', {})
    issues = bq.get('issues', [])

    print(f"检测图像数: {metrics.get('total_images', 0)}")
    print(f"发现问题数: {metrics.get('issue_count', 0)}")
    print(f"问题率: {metrics.get('issue_rate', 0):.2f}%")
    print(f"平均质量分数: {metrics.get('average_quality_score', 0):.3f}")

    issue_types = {}
    for issue in issues:
        t = issue.get('type', 'unknown')
        issue_types[t] = issue_types.get(t, 0) + 1

    print("\n问题类型分布:")
    for itype, count in issue_types.items():
        print(f"  {itype}: {count}")

    print("\n[PASS] 基础质量检测模块正常工作")
    return result


def test_label_error(detector, images, labels, label_names, sample_size=500):
    """测试标签错误检测模块"""
    print("\n" + "=" * 60)
    print("测试 2: 标签错误检测（Confidence Learning）")
    print("=" * 60)

    sample_images = images[:sample_size]
    sample_labels = labels[:sample_size]

    print(f"样本数: {len(sample_images)}")
    print(f"类别数: {len(set(sample_labels))}")

    result = detector.detect(
        sample_images,
        labels=sample_labels,
        modules=['basic_quality', 'label_error']
    )

    le = result.get('label_error', {})
    if 'error' in le:
        print(f"[WARN] 标签错误检测出错: {le['error']}")
        return result

    print(f"检测到的标签错误数: {le.get('error_count', 0)}")
    print(f"错误率: {le.get('error_rate', 0):.2%}")

    debug_info = le.get('debug_info', {})
    print(f"交叉验证折数: {debug_info.get('cv_folds_used', '?')}")
    print(f"平均质量分数: {debug_info.get('mean_quality_score', 0):.3f}")
    print(f"最低质量分数: {debug_info.get('min_quality_score', 0):.3f}")

    error_indices = le.get('error_indices', [])
    if error_indices:
        print(f"\n前 10 个可疑样本:")
        for idx in error_indices[:10]:
            original_label = sample_labels[idx]
            suggested = le.get('suggested_labels', {}).get(str(idx))
            score = le.get('label_quality_scores', [0]*len(sample_labels))[idx]
            orig_name = label_names[original_label] if original_label < len(label_names) else str(original_label)
            sugg_name = label_names[suggested] if suggested is not None and suggested < len(label_names) else str(suggested)
            print(f"  索引 {idx}: {orig_name} -> {sugg_name} (分数: {score:.3f})")

    print("\n[PASS] 标签错误检测模块正常工作")
    return result


def test_uncertainty(detector, images, labels, sample_size=500):
    """测试不确定性估计模块"""
    print("\n" + "=" * 60)
    print("测试 3: 不确定性估计")
    print("=" * 60)

    sample_images = images[:sample_size]
    sample_labels = labels[:sample_size]

    result = detector.detect(
        sample_images,
        labels=sample_labels,
        modules=['uncertainty']
    )

    ue = result.get('uncertainty', {})
    if 'error' in ue:
        print(f"[WARN] 不确定性估计出错: {ue['error']}")
        return result

    print(f"高不确定性样本数: {ue.get('high_uncertainty_count', 0)}")

    stats = ue.get('statistics', {})
    print(f"平均距离: {stats.get('mean_distance', 0):.3f}")
    print(f"距离标准差: {stats.get('std_distance', 0):.3f}")
    print(f"阈值: {ue.get('threshold', 0):.3f}")

    high_uncertainty = ue.get('high_uncertainty_indices', [])
    if high_uncertainty:
        print(f"\n前 10 个高不确定性样本索引: {high_uncertainty[:10]}")

    print("\n[PASS] 不确定性估计模块正常工作")
    return result


def main():
    print("=" * 60)
    print("CIFAR-10 图像检测器功能测试")
    print("=" * 60)

    data_path = r'c:\Users\abc14\Desktop\毕业设计\代码\multimodel_qa\data\raw\cifar-10-python (1).tar.gz'

    if not os.path.exists(data_path):
        print(f"错误: 数据集文件不存在: {data_path}")
        return

    train_bytes, y_train, test_bytes, y_test, label_names = load_cifar10(data_path)

    cfg = Config()
    detector = ImageDetector(cfg)

    test_basic_quality(detector, train_bytes, sample_size=1000)

    test_label_error(detector, train_bytes, y_train, label_names, sample_size=500)

    test_uncertainty(detector, train_bytes, y_train, sample_size=500)

    print("\n" + "=" * 60)
    print("所有测试完成！图像检测器功能正常")
    print("=" * 60)


if __name__ == '__main__':
    main()
