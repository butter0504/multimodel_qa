import os
import sys
import tarfile
import pickle
import zipfile
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
import torch

DATA_DIR = r'c:\Users\abc14\Desktop\毕业设计\代码\multimodel_qa\data\raw'
OUTPUT_DIR = r'c:\Users\abc14\Desktop\毕业设计\代码\multimodel_qa\data\uploads'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输出目录: {OUTPUT_DIR}")

def convert_cifar10n(sample_size=5000):
    print("\n" + "=" * 60)
    print("转换 CIFAR-10N 数据集")
    print("=" * 60)

    extract_dir = os.path.join(DATA_DIR, 'cifar-10-batches-py')

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
            if (i + 1) % 1000 == 0:
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


def convert_cifar100n(sample_size=5000):
    print("\n" + "=" * 60)
    print("转换 CIFAR-100N 数据集")
    print("=" * 60)

    extract_dir = os.path.join(DATA_DIR, 'cifar-100-python')

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
            if (i + 1) % 1000 == 0:
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


if __name__ == '__main__':
    print("开始转换数据集...")
    convert_cifar10n(5000)
    convert_cifar100n(5000)
    print("\n所有转换完成!")
