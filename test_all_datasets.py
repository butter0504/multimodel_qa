"""
多数据集图像检测器测试脚本
===========================
测试 data/raw 目录下所有图像数据集，验证图像检测器功能。

支持的数据集：
    1. CIFAR-10 (32x32, 10类)
    2. CIFAR-100 (32x32, 100类)
    3. Tiny-ImageNet-200 (64x64, 200类)

测试内容：
    - 基础质量检测
    - 标签错误检测
    - 不确定性估计

使用方式：
    python test_all_datasets.py
"""

import sys
import os
import tarfile
import zipfile
import pickle
import numpy as np
from io import BytesIO
from PIL import Image
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import Config
from modules.image_detector import ImageDetector

DATA_DIR = r'c:\Users\abc14\Desktop\毕业设计\代码\multimodel_qa\data\raw'


def load_cifar10(data_dir):
    """加载 CIFAR-10 数据集"""
    tar_path = os.path.join(data_dir, 'cifar-10-python (1).tar.gz')
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')

    if not os.path.exists(cifar_dir):
        print(f"解压 CIFAR-10...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)

    def load_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data']
            Y = datadict[b'labels']
            X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            return X, Y

    xs, ys = [], []
    for b in range(1, 6):
        X, Y = load_batch(os.path.join(cifar_dir, f'data_batch_{b}'))
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    with open(os.path.join(cifar_dir, 'batches.meta'), 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        label_names = [name.decode('utf-8') for name in meta[b'label_names']]

    return X_train, y_train, label_names, (32, 32)


def load_cifar100(data_dir):
    """加载 CIFAR-100 数据集"""
    tar_path = os.path.join(data_dir, 'cifar-100-python.tar.gz')
    extract_dir = os.path.join(data_dir, 'cifar-100-python')

    if not os.path.exists(extract_dir):
        print(f"解压 CIFAR-100...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)

    with open(os.path.join(extract_dir, 'train'), 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X_train = datadict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_train = np.array(datadict[b'fine_labels'])

    with open(os.path.join(extract_dir, 'meta'), 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]

    return X_train, y_train, label_names, (32, 32)


def load_tiny_imagenet(data_dir):
    """加载 Tiny-ImageNet-200 数据集"""
    zip_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')
    extract_dir = os.path.join(data_dir, 'tiny-imagenet-200')

    if not os.path.exists(extract_dir):
        print(f"解压 Tiny-ImageNet-200...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(path=data_dir)

    train_dir = os.path.join(extract_dir, 'train')
    images = []
    labels = []
    label_names = []
    label_map = {}

    class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    for class_idx, class_name in enumerate(class_dirs[:50]):
        label_map[class_name] = class_idx
        label_names.append(class_name)
        class_path = os.path.join(train_dir, class_name, 'images')
        if not os.path.exists(class_path):
            continue
        img_files = [f for f in os.listdir(class_path) if f.endswith('.JPEG')][:20]
        for img_file in img_files:
            try:
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('RGB')
                images.append(np.array(img))
                labels.append(class_idx)
            except Exception:
                continue

    return np.array(images), np.array(labels), label_names, (64, 64)


def images_to_bytes(images):
    """将 numpy 图像数组转换为字节列表"""
    bytes_list = []
    for img_array in images:
        img = Image.fromarray(img_array)
        buf = BytesIO()
        img.save(buf, format='PNG')
        bytes_list.append(buf.getvalue())
    return bytes_list


def test_dataset(name, images, labels, label_names, img_size, detector, sample_size=500):
    """测试单个数据集"""
    print("\n" + "=" * 70)
    print(f"测试数据集: {name}")
    print("=" * 70)
    print(f"图像尺寸: {img_size[0]}x{img_size[1]}")
    print(f"总图像数: {len(images)}")
    print(f"类别数: {len(label_names)}")
    print(f"测试样本数: {sample_size}")

    sample_images = images[:sample_size]
    sample_labels = labels[:sample_size]
    image_bytes = images_to_bytes(sample_images)

    start_time = time.time()

    result = detector.detect(
        image_bytes,
        labels=sample_labels,
        modules=['basic_quality', 'label_error', 'uncertainty']
    )

    elapsed = time.time() - start_time
    print(f"\n检测耗时: {elapsed:.2f} 秒")

    bq = result.get('basic_quality', {})
    metrics = bq.get('metrics', {})
    issues = bq.get('issues', [])

    print(f"\n--- 基础质量检测 ---")
    print(f"发现问题数: {metrics.get('issue_count', 0)}")
    print(f"平均质量分数: {metrics.get('average_quality_score', 0):.3f}")

    issue_types = {}
    for issue in issues:
        t = issue.get('type', 'unknown')
        issue_types[t] = issue_types.get(t, 0) + 1

    print("问题类型分布:")
    for itype, count in sorted(issue_types.items(), key=lambda x: -x[1]):
        print(f"  {itype}: {count} ({count/sample_size*100:.1f}%)")

    le = result.get('label_error', {})
    print(f"\n--- 标签错误检测 ---")
    if 'error' in le:
        print(f"[WARN] {le['error']}")
    else:
        print(f"检测到的可疑标签: {le.get('error_count', 0)}")
        print(f"平均质量分数: {le.get('debug_info', {}).get('mean_quality_score', 0):.3f}")

    ue = result.get('uncertainty', {})
    print(f"\n--- 不确定性估计 ---")
    if 'error' in ue:
        print(f"[WARN] {ue['error']}")
    else:
        print(f"高不确定性样本数: {ue.get('high_uncertainty_count', 0)}")
        print(f"平均距离: {ue.get('statistics', {}).get('mean_distance', 0):.2f}")

    return {
        'name': name,
        'img_size': img_size,
        'sample_size': sample_size,
        'elapsed': elapsed,
        'quality_score': metrics.get('average_quality_score', 0),
        'issue_count': metrics.get('issue_count', 0),
        'issue_types': issue_types,
        'label_error_count': le.get('error_count', 0) if 'error' not in le else 0,
        'uncertainty_count': ue.get('high_uncertainty_count', 0) if 'error' not in ue else 0,
    }


def main():
    print("=" * 70)
    print("多数据集图像检测器功能测试")
    print("=" * 70)

    cfg = Config()
    detector = ImageDetector(cfg)

    results = []

    datasets = [
        ('CIFAR-10', load_cifar10),
        ('CIFAR-100', load_cifar100),
        ('Tiny-ImageNet-200', load_tiny_imagenet),
    ]

    for name, loader in datasets:
        try:
            print(f"\n加载 {name} 数据集...")
            images, labels, label_names, img_size = loader(DATA_DIR)
            print(f"加载完成: {len(images)} 张图像, {len(label_names)} 个类别")

            result = test_dataset(name, images, labels, label_names, img_size, detector, sample_size=500)
            results.append(result)

        except FileNotFoundError as e:
            print(f"[SKIP] {name} 数据集文件不存在: {e}")
        except Exception as e:
            print(f"[ERROR] {name} 测试失败: {e}")

    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    print(f"\n{'数据集':<20} {'尺寸':<10} {'质量分数':<10} {'问题数':<10} {'耗时':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['img_size'][0]}x{r['img_size'][1]:<5} "
              f"{r['quality_score']:<10.3f} {r['issue_count']:<10} {r['elapsed']:<10.2f}s")

    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
