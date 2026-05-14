import numpy as np
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from .config import Config


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, data: Any) -> np.ndarray:
        pass

    @abstractmethod
    def modality(self) -> str:
        pass


class ImageEncoder(BaseEncoder):
    """
    图像深度特征提取器
    ==================
    使用 ResNet-50 预训练模型提取 2048 维特征向量。

    特点：
        - 预训练权重：ImageNet 上的 ResNet-50 (torchvision.models.resnet50)
        - 去掉最后的全连接层，输出 2048 维特征
        - 图像预处理：Resize(256) → CenterCrop(224) → ToTensor → Normalize
        - 批量提取 + GPU 加速
        - 支持文件路径 (str) 和原始字节数据 (bytes) 两种输入
        - 当 GPU/模型不可用时自动回退到统计特征
    """

    FEATURE_DIM = 2048

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.get('feature_extraction.image_model', 'resnet50')
        self.batch_size = cfg.get('feature_extraction.batch_size', 32)
        self._model = None
        self._device = None
        self._transform = None

    def modality(self) -> str:
        return 'image'

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if self.model_name == 'resnet50':
                try:
                    from torchvision.models import ResNet50_Weights
                    base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                except ImportError:
                    base = models.resnet50(pretrained=True)
            elif self.model_name == 'resnet18':
                try:
                    from torchvision.models import ResNet18_Weights
                    base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                    self.FEATURE_DIM = 512
                except ImportError:
                    base = models.resnet18(pretrained=True)
                    self.FEATURE_DIM = 512
            else:
                try:
                    from torchvision.models import ResNet50_Weights
                    base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                except ImportError:
                    base = models.resnet50(pretrained=True)

            self._model = torch.nn.Sequential(*list(base.children())[:-1])
            self._model = self._model.to(self._device)
            self._model.eval()

            self._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        except ImportError:
            self._model = None

    def extract(self, image_paths: List[Union[str, bytes]],
                batch_size: int = 32) -> np.ndarray:
        """
        批量提取图像深度特征，使用 GPU 加速。

        Parameters
        ----------
        image_paths : List[Union[str, bytes]]
            图像路径列表或原始字节数据列表，支持混合输入：
            - str: 图像文件路径，直接从磁盘读取
            - bytes: 图像原始字节数据，从内存解码
        batch_size : int
            批处理大小，默认 32。增大可提高 GPU 利用率，但需更多显存。

        Returns
        -------
        np.ndarray
            形状为 (n_images, 2048) 的特征矩阵。
            解码失败的图像对应位置为零向量。
        """
        self._load_model()
        if self._model is not None:
            return self._extract_with_model(image_paths, batch_size)
        return self._encode_with_stats(image_paths)

    def _extract_with_model(self, image_paths: List[Union[str, bytes]],
                            batch_size: int) -> np.ndarray:
        import torch
        from PIL import Image
        import io

        all_features = []
        n_images = len(image_paths)

        with torch.no_grad():
            for start in range(0, n_images, batch_size):
                batch_paths = image_paths[start:start + batch_size]
                batch_tensors = []

                for item in batch_paths:
                    try:
                        if isinstance(item, str):
                            img = Image.open(item).convert('RGB')
                        elif isinstance(item, bytes):
                            img = Image.open(io.BytesIO(item)).convert('RGB')
                        else:
                            img = Image.open(io.BytesIO(item)).convert('RGB')
                        tensor = self._transform(img)
                        batch_tensors.append(tensor)
                    except Exception:
                        batch_tensors.append(torch.zeros(3, 224, 224))

                if not batch_tensors:
                    continue

                batch = torch.stack(batch_tensors).to(self._device)
                feats = self._model(batch)
                feats = feats.squeeze(-1).squeeze(-1).cpu().numpy()
                all_features.append(feats)

        if all_features:
            return np.vstack(all_features)
        return np.zeros((n_images, self.FEATURE_DIM))

    def encode(self, data: List[bytes]) -> np.ndarray:
        self._load_model()
        if self._model is not None:
            return self._extract_with_model(data, self.batch_size)
        return self._encode_with_stats(data)

    def _encode_with_stats(self, data: List[Union[str, bytes]]) -> np.ndarray:
        from PIL import Image
        import io
        import cv2

        features = []
        for item in data:
            try:
                if isinstance(item, str):
                    img = Image.open(item)
                elif isinstance(item, bytes):
                    img = Image.open(io.BytesIO(item))
                else:
                    img = Image.open(io.BytesIO(item))
                img_array = np.array(img)
                feat = self._extract_image_stats(img_array)
                features.append(feat)
            except Exception:
                features.append(np.zeros(10))

        max_len = max(len(f) for f in features)
        padded = [np.pad(f, (0, max_len - len(f))) for f in features]
        return np.array(padded)

    def _extract_image_stats(self, img_array: np.ndarray) -> np.ndarray:
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        h, w = gray.shape[:2]
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if len(img_array.shape) == 3:
            r_mean = float(np.mean(img_array[:, :, 0]))
            g_mean = float(np.mean(img_array[:, :, 1]))
            b_mean = float(np.mean(img_array[:, :, 2]))
            r_std = float(np.std(img_array[:, :, 0]))
            g_std = float(np.std(img_array[:, :, 1]))
            b_std = float(np.std(img_array[:, :, 2]))
            dark_ratio = float(np.sum(gray < 50) / gray.size)
            bright_ratio = float(np.sum(gray > 205) / gray.size)
            return np.array([h, w, mean_val, std_val, laplacian_var,
                             r_mean, g_mean, b_mean, r_std, g_std, b_std,
                             dark_ratio, bright_ratio])
        else:
            dark_ratio = float(np.sum(gray < 50) / gray.size)
            bright_ratio = float(np.sum(gray > 205) / gray.size)
            return np.array([h, w, mean_val, std_val, laplacian_var,
                             dark_ratio, bright_ratio])


class TextEncoder(BaseEncoder):
    """
    文本深度特征提取器
    ==================
    使用 BERT-base-uncased 预训练模型提取 768 维特征向量。

    特点：
        - 预训练权重：bert-base-uncased (transformers 库)
        - 提取 [CLS] token 的 768 维向量作为句子表示
        - 批量编码 + GPU 加速
        - padding 和 truncation 到最大 512 长度
        - 当 GPU/模型不可用时自动回退到 TF-IDF
    """

    FEATURE_DIM = 768
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 512

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.get('feature_extraction.text_model', 'bert')
        self.max_features = cfg.get('feature_extraction.max_features_tfidf', 10000)
        self.batch_size = cfg.get('feature_extraction.batch_size', 32)
        self._tokenizer = None
        self._model = None
        self._device = None

    def modality(self) -> str:
        return 'text'

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModel.from_pretrained(self.MODEL_NAME)
            self._model = self._model.to(self._device)
            self._model.eval()
        except ImportError:
            self._model = None
            self._tokenizer = None

    def extract(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量提取文本深度特征，使用 GPU 加速。

        Parameters
        ----------
        texts : List[str]
            文本列表
        batch_size : int
            批处理大小，默认 32。增大可提高 GPU 利用率，但需更多显存。

        Returns
        -------
        np.ndarray
            形状为 (n_texts, 768) 的特征矩阵。
            编码失败的文本对应位置为零向量。
        """
        self._load_model()
        if self._model is not None and self._tokenizer is not None:
            return self._extract_with_bert(texts, batch_size)
        return self._encode_with_tfidf(texts)

    def _extract_with_bert(self, texts: List[str], batch_size: int) -> np.ndarray:
        import torch

        all_embeddings = []
        n_texts = len(texts)

        with torch.no_grad():
            for start in range(0, n_texts, batch_size):
                batch_texts = texts[start:start + batch_size]

                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    return_tensors='pt',
                )

                encoded = {k: v.to(self._device) for k, v in encoded.items()}

                try:
                    outputs = self._model(**encoded)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embeddings.append(cls_embeddings)
                except Exception:
                    n_batch = len(batch_texts)
                    all_embeddings.append(np.zeros((n_batch, self.FEATURE_DIM)))

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.zeros((n_texts, self.FEATURE_DIM))

    def encode(self, data: List[str]) -> np.ndarray:
        if self.model_name == 'bert':
            return self.extract(data, self.batch_size)
        return self._encode_with_tfidf(data)

    def _encode_with_tfidf(self, data: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english'
        )
        try:
            tfidf_matrix = self._vectorizer.fit_transform(data)
            return tfidf_matrix.toarray()
        except ValueError:
            return np.zeros((len(data), self.max_features))


class TabularEncoder(BaseEncoder):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.method = cfg.get('feature_extraction.tabular_method', 'standard_scaler')

    def modality(self) -> str:
        return 'tabular'

    def encode(self, data) -> np.ndarray:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        if isinstance(data, pd.DataFrame):
            X = data.copy()
        else:
            X = pd.DataFrame(data)

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].fillna('missing').astype(str)
            X[col] = le.fit_transform(X[col])

        numeric_cols = X.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].mean() if len(X[col].dropna()) > 0 else 0)

        if self.method == 'standard_scaler':
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X.values.astype(np.float64)


class FeatureExtractor:
    """多模态统一特征提取器，借鉴 Docta 的统一 embedding 处理思路"""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self._encoders = {
            'image': ImageEncoder(self.cfg),
            'text': TextEncoder(self.cfg),
            'tabular': TabularEncoder(self.cfg),
        }

    def extract_features(self, data: Any, modality: str) -> np.ndarray:
        modality = modality.lower()
        if modality not in self._encoders:
            raise ValueError(f"不支持的模态类型: {modality}，支持: {list(self._encoders.keys())}")
        return self._encoders[modality].encode(data)

    def detect_modality(self, data: Any) -> str:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return 'tabular'
        if isinstance(data, list):
            if len(data) == 0:
                return 'tabular'
            if isinstance(data[0], bytes):
                return 'image'
            if isinstance(data[0], str):
                return 'text'
        return 'tabular'

    def auto_extract(self, data: Any) -> tuple:
        modality = self.detect_modality(data)
        features = self.extract_features(data, modality)
        return features, modality

    def get_encoder(self, modality: str) -> BaseEncoder:
        return self._encoders.get(modality.lower())
