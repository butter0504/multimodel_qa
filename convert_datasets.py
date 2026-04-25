"""
数据集转换脚本
==============
将 CIFAR-10N 和 CIFAR-100N 数据集转换为 ZIP+CSV 格式，
方便通过 Web 界面上传使用。

输出格式：
    - images.zip: 包含所有图像文件
    - labels.csv: 包含 filename, label, clean_label, label_name 列

使用方式：
    python convert_datasets.py --dataset CIFAR-10N --sample_size 1000
"""

import os
import sys
import tarfile
import pickle
import zipfile
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = r'c:\Users\abc14\Desktop\毕业设计\代码\multimodel_qa\data\raw'
OUTPUT_DIR = r'c:\Users\abc14\Desktop\毕业设计\代码\multimodel_qa\data\uploads'


def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")


def convert_cifar10n(sample_size=None):
    """
    转换 CIFAR-10N 数据集为 ZIP+CSV 格式。

    Parameters
    ----------
    sample_size : int
        转换样本数，None 表示全部
    """
    print("\n" + "=" * 60)
    print("转换 CIFAR-10N 数据集")
    print("=" * 60)

    tar_path = os.path.join(DATA_DIR, 'cifar-10-python (1).tar.gz')
    extract_dir = os.path.join(DATA_DIR, 'cifar-10-batches-py')

    if not os.path.exists(extract_dir):
        print("解压 CIFAR-10...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=DATA_DIR)

    def load_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            return X

    print("加载图像数据...")
    xs = []
    for b in range(1, 6):
        X = load_batch(os.path.join(extract_dir, f'data_batch_{b}'))
        xs.append(X)
    X_train = np.concatenate(xs)

    with open(os.path.join(extract_dir, 'batches.meta'), 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        label_names = [name.decode('utf-8') for name in meta[b'label_names']]

    print("加载噪声标签...")
    noise_file = os.path.join(DATA_DIR, 'CIFAR-10_human.pt')
    noise_data = torch.load(noise_file, weights_only=False)

    noisy_labels = noise_data['worse_label']
    clean_labels = noise_data['clean_label']

    if hasattr(noisy_labels, 'tolist'):
        noisy_labels = noisy_labels.tolist()
    if hasattr(clean_labels, 'tolist'):
        clean_labels = clean_labels.tolist()

    if sample_size:
        X_train = X_train[:sample_size]
        noisy_labels = noisy_labels[:sample_size]
        clean_labels = clean_labels[:sample_size]

    total = len(X_train)
    noise_count = sum(1 for n, c in zip(noisy_labels, clean_labels) if n != c)
    print(f"样本数: {total}")
    print(f"噪声率: {noise_count/total:.1%}")

    zip_path = os.path.join(OUTPUT_DIR, 'cifar10n_images.zip')
    csv_path = os.path.join(OUTPUT_DIR, 'cifar10n_labels.csv')

    print(f"\n创建 ZIP 文件: {zip_path}")
    labels_data = []

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, img_array in enumerate(X_train):
            if (i + 1) % 5000 == 0:
                print(f"  处理进度: {i+1}/{total}")

            filename = f"cifar10n_{i:05d}.png"

            img = Image.fromarray(img_array)
            buf = BytesIO()
            img.save(buf, format='PNG')
            zf.writestr(filename, buf.getvalue())

            noisy_label = noisy_labels[i]
            clean_label = clean_labels[i]
            noisy_name = label_names[noisy_label]
            clean_name = label_names[clean_label]

            labels_data.append({
                'filename': filename,
                'label': noisy_label,
                'label_name': noisy_name,
                'clean_label': clean_label,
                'clean_label_name': clean_name,
                'is_error': noisy_label != clean_label
            })

    print(f"\n创建 CSV 文件: {csv_path}")
    df = pd.DataFrame(labels_data)
    df.to_csv(csv_path, index=False)

    print(f"\n转换完成!")
    print(f"  ZIP 文件: {zip_path}")
    print(f"  CSV 文件: {csv_path}")
    print(f"  图像数: {total}")
    print(f"  噪声率: {noise_count/total:.1%}")

    return zip_path, csv_path


def convert_cifar100n(sample_size=None):
    """
    转换 CIFAR-100N 数据集为 ZIP+CSV 格式。

    Parameters
    ----------
    sample_size : int
        转换样本数，None 表示全部
    """
    print("\n" + "=" * 60)
    print("转换 CIFAR-100N 数据集")
    print("=" * 60)

    tar_path = os.path.join(DATA_DIR, 'cifar-100-python.tar.gz')
    extract_dir = os.path.join(DATA_DIR, 'cifar-100-python')

    if not os.path.exists(extract_dir):
        print("解压 CIFAR-100...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=DATA_DIR)

    print("加载图像数据...")
    with open(os.path.join(extract_dir, 'train'), 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X_train = datadict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    with open(os.path.join(extract_dir, 'meta'), 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]

    print("加载噪声标签...")
    noise_file = os.path.join(DATA_DIR, 'CIFAR-100_human.pt')
    noise_data = torch.load(noise_file, weights_only=False)

    noisy_labels = noise_data['noisy_label']
    clean_labels = noise_data['clean_label']

    if hasattr(noisy_labels, 'tolist'):
        noisy_labels = noisy_labels.tolist()
    if hasattr(clean_labels, 'tolist'):
        clean_labels = clean_labels.tolist()

    if sample_size:
        X_train = X_train[:sample_size]
        noisy_labels = noisy_labels[:sample_size]
        clean_labels = clean_labels[:sample_size]

    total = len(X_train)
    noise_count = sum(1 for n, c in zip(noisy_labels, clean_labels) if n != c)
    print(f"样本数: {total}")
    print(f"噪声率: {noise_count/total:.1%}")

    zip_path = os.path.join(OUTPUT_DIR, 'cifar100n_images.zip')
    csv_path = os.path.join(OUTPUT_DIR, 'cifar100n_labels.csv')

    print(f"\n创建 ZIP 文件: {zip_path}")
    labels_data = []

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, img_array in enumerate(X_train):
            if (i + 1) % 5000 == 0:
                print(f"  处理进度: {i+1}/{total}")

            filename = f"cifar100n_{i:05d}.png"

            img = Image.fromarray(img_array)
            buf = BytesIO()
            img.save(buf, format='PNG')
            zf.writestr(filename, buf.getvalue())

            noisy_label = noisy_labels[i]
            clean_label = clean_labels[i]
            noisy_name = label_names[noisy_label] if noisy_label < len(label_names) else str(noisy_label)
            clean_name = label_names[clean_label] if clean_label < len(label_names) else str(clean_label)

            labels_data.append({
                'filename': filename,
                'label': noisy_label,
                'label_name': noisy_name,
                'clean_label': clean_label,
                'clean_label_name': clean_name,
                'is_error': noisy_label != clean_label
            })

    print(f"\n创建 CSV 文件: {csv_path}")
    df = pd.DataFrame(labels_data)
    df.to_csv(csv_path, index=False)

    print(f"\n转换完成!")
    print(f"  ZIP 文件: {zip_path}")
    print(f"  CSV 文件: {csv_path}")
    print(f"  图像数: {total}")
    print(f"  噪声率: {noise_count/total:.1%}")

    return zip_path, csv_path


def convert_cifar10(sample_size=None):
    """
    转换 CIFAR-10 数据集为 ZIP+CSV 格式（干净数据集）。
    """
    print("\n" + "=" * 60)
    print("转换 CIFAR-10 数据集")
    print("=" * 60)

    tar_path = os.path.join(DATA_DIR, 'cifar-10-python (1).tar.gz')
    extract_dir = os.path.join(DATA_DIR, 'cifar-10-batches-py')

    if not os.path.exists(extract_dir):
        print("解压 CIFAR-10...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=DATA_DIR)

    def load_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            Y = datadict[b'labels']
            return X, Y

    print("加载图像数据...")
    xs, ys = [], []
    for b in range(1, 6):
        X, Y = load_batch(os.path.join(extract_dir, f'data_batch_{b}'))
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    with open(os.path.join(extract_dir, 'batches.meta'), 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        label_names = [name.decode('utf-8') for name in meta[b'label_names']]

    if sample_size:
        X_train = X_train[:sample_size]
        y_train = y_train[:sample_size]

    total = len(X_train)
    print(f"样本数: {total}")

    zip_path = os.path.join(OUTPUT_DIR, 'cifar10_images.zip')
    csv_path = os.path.join(OUTPUT_DIR, 'cifar10_labels.csv')

    print(f"\n创建 ZIP 文件: {zip_path}")
    labels_data = []

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, img_array in enumerate(X_train):
            if (i + 1) % 5000 == 0:
                print(f"  处理进度: {i+1}/{total}")

            filename = f"cifar10_{i:05d}.png"

            img = Image.fromarray(img_array)
            buf = BytesIO()
            img.save(buf, format='PNG')
            zf.writestr(filename, buf.getvalue())

            label = y_train[i]
            label_name = label_names[label]

            labels_data.append({
                'filename': filename,
                'label': label,
                'label_name': label_name
            })

    print(f"\n创建 CSV 文件: {csv_path}")
    df = pd.DataFrame(labels_data)
    df.to_csv(csv_path, index=False)

    print(f"\n转换完成!")
    print(f"  ZIP 文件: {zip_path}")
    print(f"  CSV 文件: {csv_path}")
    print(f"  图像数: {total}")

    return zip_path, csv_path


def main():
    parser = argparse.ArgumentParser(description='数据集转换脚本')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['CIFAR-10', 'CIFAR-10N', 'CIFAR-100N', 'all'],
                       help='要转换的数据集')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='转换样本数 (默认全部)')

    args = parser.parse_args()

    ensure_output_dir()

    if args.dataset == 'all':
        convert_cifar10(args.sample_size)
        convert_cifar10n(args.sample_size)
        convert_cifar100n(args.sample_size)
    elif args.dataset == 'CIFAR-10':
        convert_cifar10(args.sample_size)
    elif args.dataset == 'CIFAR-10N':
        convert_cifar10n(args.sample_size)
    elif args.dataset == 'CIFAR-100N':
        convert_cifar100n(args.sample_size)

    print("\n" + "=" * 60)
    print("所有转换完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
