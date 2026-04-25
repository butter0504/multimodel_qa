"""
多格式数据适配器
================
支持多种数据格式的自动识别和转换，包括：
    - 标准数据集（CIFAR-10/100, ImageNet, MNIST, AG News）
    - 文件夹结构（文件夹名=标签）
    - 标注平台格式（Label Studio, CVAT, COCO JSON）
    - 无标签数据（仅部分检测功能）

使用方式：
    from modules.data_adapter import DataAdapter

    adapter = DataAdapter()
    result = adapter.load(source_type='standard_dataset',
                          dataset_name='CIFAR-10',
                          data_path='./data/raw/')
"""

import os
import pickle
import tarfile
import zipfile
import json
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class DataAdapter:
    """
    多格式数据适配器
    ================
    统一处理多种数据格式，输出标准化的数据结构。

    输出格式：
        {
            'images': List[bytes],          # 图像字节列表
            'labels': List[Any],            # 标签列表（可选）
            'label_names': List[str],       # 标签名称列表
            'filenames': List[str],         # 文件名列表
            'metadata': Dict[str, Any],     # 元数据
            'source_type': str,             # 数据源类型
            'has_labels': bool,             # 是否有标签
        }
    """

    SUPPORTED_STANDARD_DATASETS = [
        'CIFAR-10', 'CIFAR-100', 'CIFAR-10N',
        'ImageNet', 'Tiny-ImageNet',
        'MNIST', 'Fashion-MNIST',
        'AG-News', 'IMDB', 'SST-2',
    ]

    def __init__(self):
        self.result = None

    def load(self, source_type: str, **kwargs) -> Dict[str, Any]:
        """
        加载数据源。

        Parameters
        ----------
        source_type : str
            数据源类型：
            - 'label_file': 已有标签文件（CSV/JSON）
            - 'standard_dataset': 标准数据集（自动识别）
            - 'folder_structure': 文件夹结构（文件夹名=标签）
            - 'no_label': 无标签数据

        Returns
        -------
        Dict[str, Any]
            标准化的数据结构
        """
        if source_type == 'label_file':
            return self._load_label_file(**kwargs)
        elif source_type == 'standard_dataset':
            return self._load_standard_dataset(**kwargs)
        elif source_type == 'folder_structure':
            return self._load_folder_structure(**kwargs)
        elif source_type == 'no_label':
            return self._load_no_label(**kwargs)
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")

    def _load_label_file(self, label_file=None, image_files=None,
                         image_zip=None, **kwargs) -> Dict[str, Any]:
        """
        加载已有标签文件（CSV/JSON）。

        支持格式：
            - CSV: filename, label 列
            - JSON: Label Studio 导出格式
            - COCO JSON: annotations 格式
        """
        images = []
        labels = []
        filenames = []
        label_names = []

        if image_zip:
            images, filenames = self._load_images_from_zip(image_zip)
        elif image_files:
            for f in image_files:
                images.append(f.read())
                filenames.append(f.name)

        if label_file:
            label_data = self._parse_label_file(label_file)
            label_map = label_data.get('label_map', {})
            label_names = label_data.get('label_names', [])

            labels = [label_map.get(fn, None) for fn in filenames]

        return {
            'images': images,
            'labels': labels,
            'label_names': label_names,
            'filenames': filenames,
            'metadata': {'source': 'label_file'},
            'source_type': 'label_file',
            'has_labels': len([l for l in labels if l is not None]) > 0,
        }

    def _parse_label_file(self, label_file) -> Dict[str, Any]:
        """解析标签文件"""
        content = label_file.read()
        filename = label_file.name.lower()

        if filename.endswith('.json'):
            data = json.loads(content)

            if 'annotations' in data:
                return self._parse_coco_json(data)
            else:
                return self._parse_label_studio_json(data)

        elif filename.endswith('.csv'):
            import pandas as pd
            from io import StringIO
            df = pd.read_csv(StringIO(content.decode('utf-8')))

            if 'filename' in df.columns and 'label' in df.columns:
                label_map = dict(zip(df['filename'].astype(str), df['label']))
                label_names = sorted(df['label'].unique().tolist())
                return {'label_map': label_map, 'label_names': label_names}

        return {'label_map': {}, 'label_names': []}

    def _parse_coco_json(self, data: dict) -> Dict[str, Any]:
        """解析 COCO JSON 格式"""
        label_map = {}
        label_names = []

        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        label_names = list(categories.values())

        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            cat_id = ann['category_id']
            for img in data.get('images', []):
                if img['id'] == img_id:
                    label_map[img['file_name']] = categories.get(cat_id, str(cat_id))
                    break

        return {'label_map': label_map, 'label_names': label_names}

    def _parse_label_studio_json(self, data: list) -> Dict[str, Any]:
        """解析 Label Studio JSON 格式"""
        label_map = {}
        label_names = set()

        for item in data:
            if 'data' in item and 'image' in item['data']:
                filename = os.path.basename(item['data']['image'])
                for ann in item.get('annotations', []):
                    for result in ann.get('result', []):
                        if 'value' in result and 'choices' in result['value']:
                            label = result['value']['choices'][0]
                            label_map[filename] = label
                            label_names.add(label)

        return {'label_map': label_map, 'label_names': list(label_names)}

    def _load_standard_dataset(self, dataset_name: str, data_path: str,
                               sample_size: int = None, **kwargs) -> Dict[str, Any]:
        """
        加载标准数据集。

        支持数据集：
            - CIFAR-10, CIFAR-100, CIFAR-10N, CIFAR-100N
            - Tiny-ImageNet, ImageNet
            - MNIST, Fashion-MNIST
            - AG-News, IMDB, SST-2 (文本)
        """
        dataset_name_upper = dataset_name.upper().replace('-', '').replace('_', '')

        if 'CIFAR10N' in dataset_name_upper or dataset_name_upper == 'CIFAR10N':
            return self._load_cifar10n(data_path, sample_size)
        elif 'CIFAR100N' in dataset_name_upper or dataset_name_upper == 'CIFAR100N':
            return self._load_cifar100n(data_path, sample_size)
        elif 'CIFAR10' in dataset_name_upper:
            return self._load_cifar10(data_path, sample_size)
        elif 'CIFAR100' in dataset_name_upper:
            return self._load_cifar100(data_path, sample_size)
        elif 'TINY' in dataset_name_upper or 'TINYIMAGENET' in dataset_name_upper:
            return self._load_tiny_imagenet(data_path, sample_size)
        elif 'MNIST' in dataset_name_upper:
            return self._load_mnist(data_path, sample_size, 'fashion' in dataset_name.lower())
        elif 'AGNEWS' in dataset_name_upper or 'AG' in dataset_name_upper:
            return self._load_ag_news(data_path, sample_size)
        else:
            raise ValueError(f"不支持的标准数据集: {dataset_name}")

    def _load_cifar10(self, data_path: str, sample_size: int = None) -> Dict[str, Any]:
        """加载 CIFAR-10 数据集"""
        tar_path = self._find_file(data_path, 'cifar-10', '.tar.gz')
        if not tar_path:
            raise FileNotFoundError(f"未找到 CIFAR-10 数据集文件")

        extract_dir = os.path.join(os.path.dirname(tar_path), 'cifar-10-batches-py')
        if not os.path.exists(extract_dir):
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(tar_path))

        def load_batch(filename):
            with open(filename, 'rb') as f:
                datadict = pickle.load(f, encoding='bytes')
                X = datadict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                Y = datadict[b'labels']
                return X, Y

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

        images, filenames = self._numpy_to_bytes(X_train, 'cifar10')

        return {
            'images': images,
            'labels': y_train.tolist(),
            'label_names': label_names,
            'filenames': filenames,
            'metadata': {
                'source': 'CIFAR-10',
                'total_samples': len(images),
                'image_size': (32, 32),
                'num_classes': 10,
            },
            'source_type': 'standard_dataset',
            'has_labels': True,
        }

    def _load_cifar100(self, data_path: str, sample_size: int = None) -> Dict[str, Any]:
        """加载 CIFAR-100 数据集"""
        tar_path = self._find_file(data_path, 'cifar-100', '.tar.gz')
        if not tar_path:
            raise FileNotFoundError(f"未找到 CIFAR-100 数据集文件")

        extract_dir = os.path.join(os.path.dirname(tar_path), 'cifar-100-python')
        if not os.path.exists(extract_dir):
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(tar_path))

        with open(os.path.join(extract_dir, 'train'), 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X_train = datadict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            y_train = np.array(datadict[b'fine_labels'])

        with open(os.path.join(extract_dir, 'meta'), 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]

        if sample_size:
            X_train = X_train[:sample_size]
            y_train = y_train[:sample_size]

        images, filenames = self._numpy_to_bytes(X_train, 'cifar100')

        return {
            'images': images,
            'labels': y_train.tolist(),
            'label_names': label_names,
            'filenames': filenames,
            'metadata': {
                'source': 'CIFAR-100',
                'total_samples': len(images),
                'image_size': (32, 32),
                'num_classes': 100,
            },
            'source_type': 'standard_dataset',
            'has_labels': True,
        }

    def _load_cifar10n(self, data_path: str, sample_size: int = None,
                       noise_type: str = 'worse') -> Dict[str, Any]:
        """
        加载 CIFAR-10N 噪声标签数据集。

        CIFAR-10N 包含真实标签(clean_label)和噪声标签(worse_label等)，
        用于评估标签错误检测器的性能。

        Parameters
        ----------
        data_path : str
            数据集路径
        sample_size : int
            加载样本数
        noise_type : str
            噪声标签类型：
            - 'worse': 最差噪声标签（噪声率最高，约18%）
            - 'aggre': 聚合标签
            - 'random1/2/3': 随机噪声标签

        Returns
        -------
        Dict[str, Any]
            包含 images, labels(噪声), clean_labels(真实), label_names 等
        """
        import torch

        tar_path = self._find_file(data_path, 'cifar-10', '.tar.gz')
        if not tar_path:
            raise FileNotFoundError(f"未找到 CIFAR-10 数据集文件")

        extract_dir = os.path.join(os.path.dirname(tar_path), 'cifar-10-batches-py')
        if not os.path.exists(extract_dir):
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(tar_path))

        def load_batch(filename):
            with open(filename, 'rb') as f:
                datadict = pickle.load(f, encoding='bytes')
                X = datadict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                return X

        xs = []
        for b in range(1, 6):
            X = load_batch(os.path.join(extract_dir, f'data_batch_{b}'))
            xs.append(X)
        X_train = np.concatenate(xs)

        with open(os.path.join(extract_dir, 'batches.meta'), 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            label_names = [name.decode('utf-8') for name in meta[b'label_names']]

        noise_file = self._find_file(data_path, 'CIFAR-10_human', '.pt')
        if not noise_file:
            raise FileNotFoundError(f"未找到 CIFAR-10N 噪声标签文件 (CIFAR-10_human.pt)")

        noise_data = torch.load(noise_file, weights_only=False)

        noise_key = f'{noise_type}_label' if noise_type != 'worse' else 'worse_label'
        if noise_key not in noise_data:
            noise_key = 'worse_label'

        noisy_labels = noise_data[noise_key]
        clean_labels = noise_data['clean_label']

        if hasattr(noisy_labels, 'tolist'):
            noisy_labels = noisy_labels.tolist()
        if hasattr(clean_labels, 'tolist'):
            clean_labels = clean_labels.tolist()

        if sample_size:
            X_train = X_train[:sample_size]
            noisy_labels = noisy_labels[:sample_size]
            clean_labels = clean_labels[:sample_size]

        images, filenames = self._numpy_to_bytes(X_train, 'cifar10n')

        noise_count = sum(1 for n, c in zip(noisy_labels, clean_labels) if n != c)
        noise_rate = noise_count / len(noisy_labels) if noisy_labels else 0

        return {
            'images': images,
            'labels': noisy_labels,
            'clean_labels': clean_labels,
            'label_names': label_names,
            'filenames': filenames,
            'metadata': {
                'source': 'CIFAR-10N',
                'total_samples': len(images),
                'image_size': (32, 32),
                'num_classes': 10,
                'noise_type': noise_type,
                'noise_rate': noise_rate,
                'actual_errors': noise_count,
            },
            'source_type': 'standard_dataset',
            'has_labels': True,
            'has_clean_labels': True,
        }

    def _load_cifar100n(self, data_path: str, sample_size: int = None) -> Dict[str, Any]:
        """
        加载 CIFAR-100N 噪声标签数据集。

        CIFAR-100N 包含真实标签(clean_label)和噪声标签(noisy_label)，
        用于评估标签错误检测器的性能。

        Parameters
        ----------
        data_path : str
            数据集路径
        sample_size : int
            加载样本数

        Returns
        -------
        Dict[str, Any]
            包含 images, labels(噪声), clean_labels(真实), label_names 等
        """
        import torch

        tar_path = self._find_file(data_path, 'cifar-100', '.tar.gz')
        if not tar_path:
            raise FileNotFoundError(f"未找到 CIFAR-100 数据集文件")

        extract_dir = os.path.join(os.path.dirname(tar_path), 'cifar-100-python')
        if not os.path.exists(extract_dir):
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(tar_path))

        with open(os.path.join(extract_dir, 'train'), 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X_train = datadict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        with open(os.path.join(extract_dir, 'meta'), 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]

        noise_file = self._find_file(data_path, 'CIFAR-100_human', '.pt')
        if not noise_file:
            raise FileNotFoundError(f"未找到 CIFAR-100N 噪声标签文件 (CIFAR-100_human.pt)")

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

        images, filenames = self._numpy_to_bytes(X_train, 'cifar100n')

        noise_count = sum(1 for n, c in zip(noisy_labels, clean_labels) if n != c)
        noise_rate = noise_count / len(noisy_labels) if noisy_labels else 0

        return {
            'images': images,
            'labels': noisy_labels,
            'clean_labels': clean_labels,
            'label_names': label_names,
            'filenames': filenames,
            'metadata': {
                'source': 'CIFAR-100N',
                'total_samples': len(images),
                'image_size': (32, 32),
                'num_classes': 100,
                'noise_rate': noise_rate,
                'actual_errors': noise_count,
            },
            'source_type': 'standard_dataset',
            'has_labels': True,
            'has_clean_labels': True,
        }

    def _load_tiny_imagenet(self, data_path: str, sample_size: int = None) -> Dict[str, Any]:
        """加载 Tiny-ImageNet 数据集"""
        zip_path = self._find_file(data_path, 'tiny-imagenet', '.zip')
        if not zip_path:
            raise FileNotFoundError(f"未找到 Tiny-ImageNet 数据集文件")

        extract_dir = os.path.join(os.path.dirname(zip_path), 'tiny-imagenet-200')
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(path=os.path.dirname(zip_path))

        train_dir = os.path.join(extract_dir, 'train')
        images = []
        labels = []
        filenames = []
        label_names = []
        label_map = {}

        class_dirs = sorted([d for d in os.listdir(train_dir)
                           if os.path.isdir(os.path.join(train_dir, d))])

        for class_idx, class_name in enumerate(class_dirs):
            label_map[class_name] = class_idx
            label_names.append(class_name)
            class_path = os.path.join(train_dir, class_name, 'images')

            if not os.path.exists(class_path):
                continue

            img_files = sorted([f for f in os.listdir(class_path) if f.endswith('.JPEG')])
            for img_file in img_files:
                try:
                    img_path = os.path.join(class_path, img_file)
                    img = Image.open(img_path).convert('RGB')
                    buf = BytesIO()
                    img.save(buf, format='PNG')
                    images.append(buf.getvalue())
                    labels.append(class_idx)
                    filenames.append(f"{class_name}/{img_file}")
                except Exception:
                    continue

        if sample_size and len(images) > sample_size:
            images = images[:sample_size]
            labels = labels[:sample_size]
            filenames = filenames[:sample_size]

        return {
            'images': images,
            'labels': labels,
            'label_names': label_names,
            'filenames': filenames,
            'metadata': {
                'source': 'Tiny-ImageNet-200',
                'total_samples': len(images),
                'image_size': (64, 64),
                'num_classes': len(label_names),
            },
            'source_type': 'standard_dataset',
            'has_labels': True,
        }

    def _load_mnist(self, data_path: str, sample_size: int = None,
                    fashion: bool = False) -> Dict[str, Any]:
        """加载 MNIST/Fashion-MNIST 数据集"""
        try:
            from torchvision import datasets

            name = 'FashionMNIST' if fashion else 'MNIST'
            dataset = datasets.MNIST(data_path, train=True, download=True) if not fashion \
                     else datasets.FashionMNIST(data_path, train=True, download=True)

            X_train = dataset.data.numpy()
            y_train = dataset.targets.numpy()

            if fashion:
                label_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            else:
                label_names = [str(i) for i in range(10)]

            if sample_size:
                X_train = X_train[:sample_size]
                y_train = y_train[:sample_size]

            images = []
            filenames = []
            for i, img_array in enumerate(X_train):
                img = Image.fromarray(img_array.numpy() if hasattr(img_array, 'numpy') else img_array)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                buf = BytesIO()
                img.save(buf, format='PNG')
                images.append(buf.getvalue())
                filenames.append(f"mnist_{i:05d}.png")

            return {
                'images': images,
                'labels': y_train.tolist(),
                'label_names': label_names,
                'filenames': filenames,
                'metadata': {
                    'source': name,
                    'total_samples': len(images),
                    'image_size': (28, 28),
                    'num_classes': 10,
                },
                'source_type': 'standard_dataset',
                'has_labels': True,
            }
        except ImportError:
            raise ImportError("请安装 torchvision: pip install torchvision")

    def _load_ag_news(self, data_path: str, sample_size: int = None) -> Dict[str, Any]:
        """加载 AG News 文本数据集"""
        try:
            from torchtext.datasets import AG_NEWS

            train_iter = AG_NEWS(root=data_path, split='train')
            texts = []
            labels = []
            label_names = ['World', 'Sports', 'Business', 'Sci/Tech']

            for label, text in train_iter:
                texts.append(text)
                labels.append(label - 1)
                if sample_size and len(texts) >= sample_size:
                    break

            return {
                'texts': texts,
                'labels': labels,
                'label_names': label_names,
                'filenames': [f"agnews_{i:05d}" for i in range(len(texts))],
                'metadata': {
                    'source': 'AG-News',
                    'total_samples': len(texts),
                    'num_classes': 4,
                    'modality': 'text',
                },
                'source_type': 'standard_dataset',
                'has_labels': True,
            }
        except ImportError:
            raise ImportError("请安装 torchtext: pip install torchtext")

    def _load_folder_structure(self, folder_path: str, **kwargs) -> Dict[str, Any]:
        """
        加载文件夹结构数据（文件夹名=标签）。

        文件夹结构示例：
            dataset/
                cat/
                    img001.jpg
                    img002.jpg
                dog/
                    img003.jpg
                    img004.jpg
        """
        images = []
        labels = []
        filenames = []
        label_names = []
        label_map = {}

        supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        class_dirs = sorted([d for d in os.listdir(folder_path)
                           if os.path.isdir(os.path.join(folder_path, d))])

        for class_idx, class_name in enumerate(class_dirs):
            label_map[class_name] = class_idx
            label_names.append(class_name)
            class_path = os.path.join(folder_path, class_name)

            for fname in sorted(os.listdir(class_path)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in supported_ext:
                    try:
                        fpath = os.path.join(class_path, fname)
                        with open(fpath, 'rb') as f:
                            images.append(f.read())
                        labels.append(class_idx)
                        filenames.append(f"{class_name}/{fname}")
                    except Exception:
                        continue

        return {
            'images': images,
            'labels': labels,
            'label_names': label_names,
            'filenames': filenames,
            'metadata': {
                'source': 'folder_structure',
                'total_samples': len(images),
                'num_classes': len(label_names),
            },
            'source_type': 'folder_structure',
            'has_labels': True,
        }

    def _load_no_label(self, image_zip=None, image_files=None,
                       image_folder=None, **kwargs) -> Dict[str, Any]:
        """
        加载无标签数据（仅部分检测功能）。
        """
        images = []
        filenames = []

        if image_zip:
            images, filenames = self._load_images_from_zip(image_zip)
        elif image_folder:
            images, filenames = self._load_images_from_folder(image_folder)
        elif image_files:
            for f in image_files:
                images.append(f.read())
                filenames.append(f.name)

        return {
            'images': images,
            'labels': None,
            'label_names': [],
            'filenames': filenames,
            'metadata': {
                'source': 'no_label',
                'total_samples': len(images),
            },
            'source_type': 'no_label',
            'has_labels': False,
        }

    def _load_images_from_zip(self, zip_bytes: bytes) -> Tuple[List[bytes], List[str]]:
        """从 ZIP 文件加载图像"""
        images = []
        filenames = []
        supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            for name in sorted(zf.namelist()):
                ext = os.path.splitext(name)[1].lower()
                if ext in supported_ext and not name.startswith('__MACOSX'):
                    try:
                        img_bytes = zf.read(name)
                        Image.open(BytesIO(img_bytes))
                        images.append(img_bytes)
                        filenames.append(os.path.basename(name))
                    except Exception:
                        continue

        return images, filenames

    def _load_images_from_folder(self, folder_path: str) -> Tuple[List[bytes], List[str]]:
        """从文件夹加载图像"""
        images = []
        filenames = []
        supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        for root, dirs, files in os.walk(folder_path):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in supported_ext:
                    try:
                        fpath = os.path.join(root, fname)
                        with open(fpath, 'rb') as f:
                            images.append(f.read())
                        filenames.append(fname)
                    except Exception:
                        continue

        return images, filenames

    def _numpy_to_bytes(self, images: np.ndarray, prefix: str) -> Tuple[List[bytes], List[str]]:
        """将 numpy 数组转换为字节列表"""
        bytes_list = []
        filenames = []

        for i, img_array in enumerate(images):
            img = Image.fromarray(img_array)
            buf = BytesIO()
            img.save(buf, format='PNG')
            bytes_list.append(buf.getvalue())
            filenames.append(f"{prefix}_{i:05d}.png")

        return bytes_list, filenames

    def _find_file(self, directory: str, pattern: str, ext: str) -> Optional[str]:
        """在目录中查找匹配的文件"""
        for f in os.listdir(directory):
            if pattern.lower() in f.lower() and f.endswith(ext):
                return os.path.join(directory, f)
        return None

    def detect_dataset_type(self, data_path: str) -> Optional[str]:
        """
        自动检测数据集类型。

        Returns
        -------
        Optional[str]
            检测到的数据集名称，如 'CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet' 等
        """
        if not os.path.exists(data_path):
            return None

        files = os.listdir(data_path)

        for f in files:
            f_lower = f.lower()
            if 'cifar-10' in f_lower or 'cifar10' in f_lower:
                return 'CIFAR-10'
            elif 'cifar-100' in f_lower or 'cifar100' in f_lower:
                return 'CIFAR-100'
            elif 'tiny-imagenet' in f_lower or 'tinyimagenet' in f_lower:
                return 'Tiny-ImageNet'
            elif 'mnist' in f_lower:
                if 'fashion' in f_lower:
                    return 'Fashion-MNIST'
                return 'MNIST'

        if os.path.isdir(data_path):
            subdirs = [d for d in os.listdir(data_path)
                      if os.path.isdir(os.path.join(data_path, d))]
            if 'cifar-10-batches-py' in subdirs:
                return 'CIFAR-10'
            elif 'cifar-100-python' in subdirs:
                return 'CIFAR-100'
            elif 'tiny-imagenet-200' in subdirs:
                return 'Tiny-ImageNet'

        return None

    def export_to_csv(self, result: Dict[str, Any], output_path: str):
        """
        将数据导出为 CSV 格式。

        CSV 格式：
            filename, label, label_name
        """
        import pandas as pd

        rows = []
        for i, fname in enumerate(result.get('filenames', [])):
            row = {'filename': fname}
            if result.get('labels') and i < len(result['labels']):
                label = result['labels'][i]
                row['label'] = label
                if result.get('label_names') and isinstance(label, int):
                    row['label_name'] = result['label_names'][label]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        return output_path
