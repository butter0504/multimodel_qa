from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
import io
from typing import Dict, Any, List, Optional

from .base_pool import BasePool
from ..config import Config


class SampleQualityPool(BasePool):
    """
    样本质量检测池
    ==============
    包含4种样本质量检测方法，输出质量异常分数。

    方法列表：
        1. laplacian_variance  - 拉普拉斯算子边缘检测方差
        2. brisque_score       - 无参考图像质量评估
        3. ssim_similarity     - 结构相似性检测重复图像
        4. clip_score          - 图文语义一致性计算
    """

    def pool_name(self) -> str:
        return "sample_quality"

    def _register_methods(self):
        self._methods = {
            "laplacian_variance": self.laplacian_variance,
            "brisque_score": self.brisque_score,
            "ssim_similarity": self.ssim_similarity,
            "clip_score": self.clip_score,
        }

    def _to_gray(self, img_bytes: bytes) -> Optional[np.ndarray]:
        try:
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img_pil)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        except Exception:
            return None

    def _to_array(self, img_bytes: bytes) -> Optional[np.ndarray]:
        try:
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return np.array(img_pil)
        except Exception:
            return None

    def laplacian_variance(self, images: List[bytes], **kwargs) -> np.ndarray:
        """
        拉普拉斯算子边缘检测方差。

        使用 Laplacian 算子计算图像的二阶导数，方差越低说明图像越模糊。
        将低方差的模糊度映射为高质量异常分数。

        Parameters
        ----------
        images : List[bytes]
            图像字节数据列表

        Returns
        -------
        np.ndarray
            质量异常分数 (n_images,)，值域 [0, 1]，越高越异常
        """
        blur_threshold = self.cfg.get('detection.blur_threshold', 50)
        scores = []

        for img_bytes in images:
            gray = self._to_gray(img_bytes)
            if gray is None:
                scores.append(1.0)
                continue

            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            anomaly = max(0.0, 1.0 - laplacian_var / (blur_threshold * 10.0))
            scores.append(float(np.clip(anomaly, 0.0, 1.0)))

        return np.array(scores, dtype=np.float64)

    def brisque_score(self, images: List[bytes], **kwargs) -> np.ndarray:
        """
        无参考图像质量评估（BRISQUE）。

        使用 OpenCV 的 BRISQUE 质量评估模型计算图像质量分数。
        BRISQUE 分数越低表示质量越好，将其归一化为异常分数。

        Parameters
        ----------
        images : List[bytes]
            图像字节数据列表

        Returns
        -------
        np.ndarray
            质量异常分数 (n_images,)，值域 [0, 1]，越高越异常
        """
        scores = []

        for img_bytes in images:
            img_array = self._to_array(img_bytes)
            if img_array is None:
                scores.append(1.0)
                continue

            try:
                brisque = cv2.quality.QualityBRISQUE_create()
                bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                quality = brisque.compute(bgr)
                brisque_score_val = float(quality[0])

                anomaly = min(brisque_score_val / 100.0, 1.0)
                scores.append(float(np.clip(anomaly, 0.0, 1.0)))
            except Exception:
                gray = self._to_gray(img_bytes)
                if gray is None:
                    scores.append(0.5)
                    continue

                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                contrast = float(np.std(gray) / 128.0)
                noise = self._estimate_noise(gray)

                anomaly = (1.0 - min(lap_var / 500.0, 1.0)) * 0.4 + \
                          (1.0 - min(contrast, 1.0)) * 0.3 + \
                          min(noise / 50.0, 1.0) * 0.3
                scores.append(float(np.clip(anomaly, 0.0, 1.0)))

        return np.array(scores, dtype=np.float64)

    def _estimate_noise(self, gray: np.ndarray) -> float:
        denoised = cv2.medianBlur(gray, 3)
        residual = gray.astype(np.float64) - denoised.astype(np.float64)
        return float(np.std(residual))

    def ssim_similarity(self, images: List[bytes],
                        threshold: float = 0.95, **kwargs) -> np.ndarray:
        """
        结构相似性检测重复图像。

        计算每对图像之间的结构相似性指数（SSIM），
        找出与其它图像高度相似的重复图像，输出重复异常分数。

        Parameters
        ----------
        images : List[bytes]
            图像字节数据列表
        threshold : float
            SSIM 阈值，高于此值视为重复

        Returns
        -------
        np.ndarray
            质量异常分数 (n_images,)，值域 [0, 1]，越高越异常（越可能是重复）
        """
        from skimage.metrics import structural_similarity as ssim

        n = len(images)
        if n < 2:
            return np.zeros(n, dtype=np.float64)

        resized = []
        target_size = (64, 64)
        for img_bytes in images:
            gray = self._to_gray(img_bytes)
            if gray is None:
                resized.append(np.zeros(target_size, dtype=np.float64))
                continue
            resized_img = cv2.resize(gray, target_size)
            resized.append(resized_img.astype(np.float64))

        max_similarity = np.zeros(n, dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    sim = ssim(resized[i], resized[j], data_range=255.0)
                    if sim > max_similarity[i]:
                        max_similarity[i] = sim
                    if sim > max_similarity[j]:
                        max_similarity[j] = sim
                except Exception:
                    continue

        anomaly = np.clip((max_similarity - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)
        return anomaly.astype(np.float64)

    def clip_score(self, images: List[bytes],
                   texts: Optional[List[str]] = None, **kwargs) -> np.ndarray:
        """
        图文语义一致性计算（CLIP Score）。

        使用 CLIP 模型计算图像与文本的语义相似度。
        相似度越低，说明图文不一致，质量异常分数越高。

        Parameters
        ----------
        images : List[bytes]
            图像字节数据列表
        texts : Optional[List[str]]
            对应的文本描述列表，若为 None 则使用空字符串

        Returns
        -------
        np.ndarray
            质量异常分数 (n_images,)，值域 [0, 1]，越高越异常
        """
        if texts is None:
            texts = [""] * len(images)

        try:
            return self._clip_score_with_model(images, texts)
        except Exception:
            return self._clip_score_fallback(images, texts)

    def _clip_score_with_model(self, images: List[bytes],
                                texts: List[str]) -> np.ndarray:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()

        scores = []
        for img_bytes, text in zip(images, texts):
            try:
                img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                if text.strip():
                    inputs = processor(text=[text], images=img_pil,
                                       return_tensors="pt", padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        similarity = outputs.logits_per_image[0, 0].item()
                        clip_sim = (similarity + 1.0) / 2.0
                else:
                    clip_sim = 0.5

                anomaly = 1.0 - float(np.clip(clip_sim, 0.0, 1.0))
                scores.append(anomaly)
            except Exception:
                scores.append(0.5)

        return np.array(scores, dtype=np.float64)

    def _clip_score_fallback(self, images: List[bytes],
                              texts: List[str]) -> np.ndarray:
        scores = []
        for img_bytes, text in zip(images, texts):
            gray = self._to_gray(img_bytes)
            if gray is None:
                scores.append(0.5)
                continue

            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            contrast = float(np.std(gray) / 128.0)
            quality_proxy = min(lap_var / 500.0, 1.0) * 0.5 + min(contrast, 1.0) * 0.5

            if text.strip():
                text_len = min(len(text) / 100.0, 1.0)
                consistency = quality_proxy * 0.7 + text_len * 0.3
            else:
                consistency = quality_proxy

            anomaly = 1.0 - consistency
            scores.append(float(np.clip(anomaly, 0.0, 1.0)))

        return np.array(scores, dtype=np.float64)
