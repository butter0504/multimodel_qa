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
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.get('feature_extraction.image_model', 'resnet50')
        self.resize = cfg.get('feature_extraction.image_resize', 224)
        self.batch_size = cfg.get('feature_extraction.batch_size', 32)
        self._model = None

    def modality(self) -> str:
        return 'image'

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torchvision.models as models
            import torch
            if self.model_name == 'resnet50':
                base = models.resnet50(pretrained=False)
            elif self.model_name == 'resnet18':
                base = models.resnet18(pretrained=False)
            else:
                base = models.resnet18(pretrained=False)
            self._model = torch.nn.Sequential(*list(base.children())[:-1])
            self._model.eval()
        except ImportError:
            self._model = None

    def encode(self, data: List[bytes]) -> np.ndarray:
        self._load_model()
        if self._model is not None:
            return self._encode_with_model(data)
        return self._encode_with_stats(data)

    def _encode_with_model(self, data: List[bytes]) -> np.ndarray:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        import io

        transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        embeddings = []
        for img_bytes in data:
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    feat = self._model(tensor)
                embeddings.append(feat.squeeze().numpy())
            except Exception:
                embeddings.append(np.zeros(512))

        return np.array(embeddings)

    def _encode_with_stats(self, data: List[bytes]) -> np.ndarray:
        from PIL import Image
        import io
        import cv2

        features = []
        for img_bytes in data:
            try:
                img = Image.open(io.BytesIO(img_bytes))
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
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.get('feature_extraction.text_model', 'tfidf')
        self.max_features = cfg.get('feature_extraction.max_features_tfidf', 10000)
        self._vectorizer = None

    def modality(self) -> str:
        return 'text'

    def encode(self, data: List[str]) -> np.ndarray:
        if self.model_name == 'bert':
            return self._encode_with_bert(data)
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

    def _encode_with_bert(self, data: List[str]) -> np.ndarray:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained('bert-base-uncased')
            model.eval()

            embeddings = []
            batch_size = self.cfg.get('feature_extraction.batch_size', 32)

            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True,
                                    max_length=512, return_tensors='pt')
                with torch.no_grad():
                    outputs = model(**encoded)
                batch_emb = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_emb)

            return np.vstack(embeddings)
        except ImportError:
            return self._encode_with_tfidf(data)


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
