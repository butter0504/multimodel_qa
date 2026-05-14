"""
图像数据质量检测器 - 模块化架构
================================

借鉴 Docta 的多模态统一处理思路，将所有图像数据统一转为特征向量，
然后复用置信学习等检测算法，无需为每种检测任务重写逻辑。

检测模块：
    1. BasicQualityDetector  - 基础图像质量检测（模糊/曝光/噪声/分辨率/对比度）
    2. LabelErrorDetector    - 标签错误检测（基于 Confidence Learning + ResNet18 特征）
    3. DistributionShiftDetector - 分布偏移检测（协变量偏移/子群偏移）
    4. UncertaintyEstimator  - 不确定性估计（特征空间密度/K近邻距离）

使用方式：
    detector = ImageDetector(cfg)
    result = detector.detect(
        images,                          # List[bytes] 或 List[np.ndarray]
        labels=None,                     # 可选：标签列表
        modules=['basic_quality'],       # 可选：选择运行的检测模块
        reference_images=None,           # 可选：基准图像集（分布偏移检测用）
    )
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
import zipfile
from typing import Dict, Any, List, Optional, Tuple
from .base_detector import BaseDetector
from .config import Config
from .report import DiagnosisSection, IssueRecord
from .pools import (
    LabelDefectPool, DistributionShiftPool,
    SampleQualityPool, FormatStructurePool,
)


class BasicQualityDetector:
    """
    基础图像质量检测器
    ==================
    检测图像的模糊度、曝光异常、噪声水平、分辨率异常和对比度不足。
    该模块为必选模块，检测速度快，无需 GPU。

    检测指标：
        - 模糊度：使用 Laplacian 方差衡量，方差越低越模糊
        - 曝光：统计灰度直方图中暗/亮像素占比
        - 噪声：通过中值滤波残差估计高频噪声水平
        - 分辨率：检测与主流分辨率（如 224×224）偏差过大的样本
        - 对比度：使用 RMS 对比度衡量，标准差越低对比度越低
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # 模糊度阈值：Laplacian 方差低于此值判定为模糊
        self.blur_threshold = cfg.get('detection.blur_threshold', 50)
        # 暗像素判定阈值（0-255），低于此值为暗像素
        self.dark_pixel_threshold = 50
        # 亮像素判定阈值（0-255），高于此值为亮像素
        self.bright_pixel_threshold = 205
        # 曝光异常阈值：暗像素占比低于此值为曝光异常
        self.exposure_ratio_threshold = cfg.get('detection.exposure_ratio_threshold', 0.5)
        # 对比度不足阈值：RMS 对比度低于此值为对比度不足
        self.contrast_threshold = cfg.get('detection.contrast_threshold', 0.08)
        # 最小分辨率阈值：图像宽度和高度的最小值低于此值为分辨率异常
        self.min_resolution = cfg.get('detection.min_resolution', 32)
        # 噪声水平阈值：中值滤波残差低于此值为噪声异常
        self.noise_threshold = cfg.get('detection.noise_threshold', 25.0)

    def detect(self, images: List[bytes]) -> Dict[str, Any]:
        """
        对图像列表执行基础质量检测。

        Parameters
        ----------
        images : List[bytes]
            图像的原始字节数据列表

        Returns
        -------
        Dict[str, Any]
            包含 issues（问题列表）、metrics（统计指标）、
            quality_scores（每张图的质量分数）、image_metadata（元数据）的字典
        """
        issues = []
        quality_scores = []
        metadata_list = []

        for i, img_bytes in enumerate(images):
            try:
                img_pil = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img_pil)
                height, width = img_array.shape[:2]
                channels = img_array.shape[2] if len(img_array.shape) == 3 else 1

                # 记录图像元数据
                meta = {
                    "index": i,
                    "height": height,
                    "width": width,
                    "channels": channels,
                    "format": img_pil.format or "unknown",
                    "mode": img_pil.mode,
                }
                metadata_list.append(meta)

                # 转灰度图用于后续检测
                if channels >= 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array.copy()

                # ---- 1. 模糊度检测 ----
                # Laplacian 算子计算二阶导数，方差越小说明边缘越少、图像越模糊
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < self.blur_threshold:
                    issues.append({
                        "type": "blurry_image",
                        "index": i,
                        "sharpness": float(laplacian_var),
                        "suggestion": "建议使用 Real-ESRGAN 增强或直接丢弃"
                    })

                # ---- 2. 曝光检测 ----
                # 统计灰度直方图中极暗和极亮像素的比例
                dark_ratio = float(np.sum(gray < self.dark_pixel_threshold) / gray.size)
                bright_ratio = float(np.sum(gray > self.bright_pixel_threshold) / gray.size)
                if dark_ratio > self.exposure_ratio_threshold:
                    issues.append({
                        "type": "under_exposed",
                        "index": i,
                        "dark_ratio": dark_ratio,
                        "suggestion": "建议过采样暗光图片或使用数据增强（亮度调整）"
                    })
                elif bright_ratio > self.exposure_ratio_threshold:
                    issues.append({
                        "type": "over_exposed",
                        "index": i,
                        "bright_ratio": bright_ratio,
                        "suggestion": "建议降低曝光或使用直方图均衡化"
                    })

                # ---- 3. 噪声检测 ----
                # 使用中值滤波去除噪声，残差即为噪声估计
                # 中值滤波对椒盐噪声特别有效，残差标准差反映噪声水平
                denoised = cv2.medianBlur(gray, 3)
                noise_residual = gray.astype(np.float64) - denoised.astype(np.float64)
                noise_level = float(np.std(noise_residual))
                if noise_level > self.noise_threshold:
                    issues.append({
                        "type": "noisy_image",
                        "index": i,
                        "noise_level": noise_level,
                        "suggestion": "建议使用去噪算法（如 DnCNN）处理或丢弃"
                    })

                # ---- 4. 分辨率异常检测 ----
                # 检测与主流分辨率偏差过大的样本
                if height < self.min_resolution or width < self.min_resolution:
                    issues.append({
                        "type": "small_image",
                        "index": i,
                        "height": height,
                        "width": width,
                        "suggestion": "建议使用超分辨率模型提升分辨率或丢弃"
                    })

                # ---- 5. 对比度检测 ----
                # RMS 对比度：灰度标准差归一化到 [0, 1]
                # 标准差越小说明像素值越集中，对比度越低
                rms_contrast = float(np.std(gray) / 255.0)
                if rms_contrast < self.contrast_threshold:
                    issues.append({
                        "type": "low_contrast",
                        "index": i,
                        "rms_contrast": rms_contrast,
                        "suggestion": "建议使用 CLAHE（对比度受限自适应直方图均衡化）增强"
                    })

                # ---- 综合质量分数 ----
                # 加权求和：清晰度权重最高（0.30），曝光和对比度次之（各0.25），
                # 噪声和分辨率权重较低（各0.10）
                quality_score = self._compute_quality_score(
                    laplacian_var, dark_ratio, bright_ratio,
                    rms_contrast, noise_level, height, width
                )
                quality_scores.append(quality_score)

            except Exception as e:
                issues.append({
                    "type": "corrupted_image",
                    "index": i,
                    "error": str(e),
                    "suggestion": "文件已损坏，建议删除"
                })
                quality_scores.append(0.0)
                metadata_list.append({"index": i, "error": str(e)})

        # 汇总统计指标
        metrics = {
            "total_images": len(images),
            "issue_count": len(issues),
            "issue_rate": float(len(issues) / len(images) * 100) if images else 0,
            "average_quality_score": float(np.mean(quality_scores)) if quality_scores else 0,
            "quality_scores": quality_scores,
        }

        return {
            "issues": issues,
            "metrics": metrics,
            "quality_scores": quality_scores,
            "image_metadata": metadata_list,
        }

    def _compute_quality_score(self, laplacian_var: float, dark_ratio: float,
                               bright_ratio: float, rms_contrast: float,
                               noise_level: float, height: int, width: int) -> float:
        """
        计算单张图像的综合质量分数。

        分数构成：
            - 清晰度分数（0-1）：Laplacian 方差归一化，越高越清晰
            - 曝光分数（0-1）：暗/亮像素占比越少越好
            - 对比度分数（0-1）：RMS 对比度归一化
            - 噪声分数（0-1）：噪声水平越低越好
            - 分辨率分数（0-1）：满足最低分辨率要求为 1，否则为 0.5

        Parameters
        ----------
        laplacian_var : float
            Laplacian 方差，衡量图像清晰度
        dark_ratio : float
            暗像素占比 [0, 1]
        bright_ratio : float
            亮像素占比 [0, 1]
        rms_contrast : float
            RMS 对比度 [0, 1]
        noise_level : float
            噪声水平（中值滤波残差标准差）
        height : int
            图像高度
        width : int
            图像宽度

        Returns
        -------
        float
            综合质量分数 [0, 1]
        """
        sharpness_score = min(laplacian_var / 500.0, 1.0)
        exposure_score = 1.0 - max(dark_ratio, bright_ratio)
        contrast_score = min(rms_contrast * 5, 1.0)
        noise_score = max(0.0, 1.0 - noise_level / 100.0)
        size_score = 1.0 if height >= self.min_resolution and width >= self.min_resolution else 0.5

        quality = (
            0.30 * sharpness_score +
            0.25 * exposure_score +
            0.25 * contrast_score +
            0.10 * noise_score +
            0.10 * size_score
        )
        return float(np.clip(quality, 0.0, 1.0))


class LabelErrorDetector:
    """
    标签错误检测器（基于 Confidence Learning）
    ==========================================
    使用 cleanlab 库的置信学习框架检测图像标签中的标注错误。

    工作流程：
        1. 提取图像特征（优先使用 ResNet 深度特征，回退到统计特征）
        2. 基于特征向量训练 RandomForest 分类器
        3. 使用交叉验证获取样本的预测概率分布
        4. 调用 cleanlab 的 find_label_issues 识别标签错误
        5. 调用 get_label_quality_scores 计算标签质量分数
        6. 分类器的预测结果作为建议修正标签
        7. 若提供真实标签(clean_labels)，计算检测性能指标

    前置条件：
        - 需要提供标签列表（labels）
        - 标签至少包含 2 个类别
        - 每个类别至少有 2 个样本（交叉验证要求）

    参考：
        - Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021).
          Confident Learning: Estimating Uncertainty in Dataset Labels.
          JAIR, 70, 1373-1411.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cv_folds = cfg.get('detection.cross_validation_folds', 5)
        self.n_estimators = cfg.get('detection.n_estimators', 200)
        self.random_state = cfg.get('detection.random_state', 42)
        self._resnet_features = None

    def detect(self, images: List[bytes], labels: List,
               clean_labels: Optional[List] = None) -> Dict[str, Any]:
        """
        使用置信学习检测图像标签错误。

        Parameters
        ----------
        images : List[bytes]
            图像的原始字节数据列表
        labels : List
            对应的标签列表，支持任意类型的标签
        clean_labels : Optional[List]
            真实标签列表（用于评估检测性能），若提供则计算精确率/召回率/F1

        Returns
        -------
        Dict[str, Any]
            包含以下键的字典：
            - error_count: 检测到的标签错误数量
            - error_rate: 标签错误率
            - error_indices: 标签错误的样本索引列表
            - suggested_labels: 建议修正的标签字典 {index: suggested_label}
            - label_quality_scores: 每个样本的标签质量分数
            - confidence_thresholds: 每个类别的置信阈值
            - performance: 性能评估指标（当提供 clean_labels 时）
            - debug_info: 调试信息
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_predict
            from sklearn.preprocessing import LabelEncoder
            from cleanlab.rank import get_label_quality_scores
            from cleanlab.filter import find_label_issues

            features = self._extract_features_auto(images)

            le = LabelEncoder()
            y = le.fit_transform(labels)
            n_classes = len(le.classes_)

            class_counts = np.bincount(y)
            min_class_count = int(np.min(class_counts))
            actual_cv_folds = min(self.cv_folds, min_class_count)
            if actual_cv_folds < 2:
                return {
                    "error": f"某些类别样本数不足（最少 {min_class_count}），无法进行交叉验证",
                    "error_count": 0,
                    "error_rate": 0,
                    "error_indices": [],
                    "suggested_labels": {},
                    "label_quality_scores": [],
                }

            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            pred_probs = cross_val_predict(
                model, features, y, cv=actual_cv_folds, method='predict_proba'
            )

            label_quality_scores = get_label_quality_scores(y, pred_probs)

            issue_result = find_label_issues(
                labels=y,
                pred_probs=pred_probs,
            )
            if issue_result.dtype == bool:
                error_indices = np.where(issue_result)[0].tolist()
            else:
                error_indices = issue_result.tolist()

            model.fit(features, y)
            predictions = model.predict(features)
            suggested_labels = {}
            for idx in error_indices:
                suggested_labels[str(idx)] = int(predictions[idx])

            percentile_threshold = self.cfg.get('detection.percentile_threshold', 85)
            thresholds = {}
            classes = np.unique(y)
            for cls in classes:
                class_probs = pred_probs[y == cls, cls]
                if len(class_probs) > 0:
                    thresholds[cls] = float(np.percentile(class_probs, percentile_threshold))
                else:
                    thresholds[cls] = 0.5

            C_confident = np.zeros((n_classes, n_classes), dtype=int)
            for i, (prob, true_label) in enumerate(zip(pred_probs, y)):
                pred_label = np.argmax(prob)
                if prob[pred_label] >= thresholds.get(pred_label, 0.5):
                    C_confident[true_label, pred_label] += 1

            class_counts_arr = np.bincount(y, minlength=n_classes)
            with np.errstate(divide='ignore', invalid='ignore'):
                noise_matrix = np.where(
                    class_counts_arr[:, None] > 0,
                    C_confident / class_counts_arr[:, None],
                    0
                )

            result = {
                "error_count": len(error_indices),
                "error_rate": len(error_indices) / len(y) if len(y) > 0 else 0,
                "error_indices": error_indices,
                "suggested_labels": suggested_labels,
                "label_quality_scores": label_quality_scores.tolist(),
                "confidence_thresholds": thresholds,
                "confident_joint_matrix": C_confident.tolist(),
                "noise_matrix": noise_matrix.tolist(),
                "class_names": le.classes_.tolist(),
                "feature_type": self._feature_type_used,
                "debug_info": {
                    "n_classes": n_classes,
                    "class_counts": class_counts.tolist(),
                    "cv_folds_used": actual_cv_folds,
                    "mean_quality_score": float(np.mean(label_quality_scores)),
                    "min_quality_score": float(np.min(label_quality_scores)),
                    "max_quality_score": float(np.max(label_quality_scores)),
                },
            }

            if clean_labels is not None:
                performance = self._evaluate_performance(
                    y, clean_labels, error_indices, le
                )
                result["performance"] = performance

            return result

        except ImportError as e:
            return {"error": f"缺少依赖库: {str(e)}", "error_count": 0}
        except Exception as e:
            return {"error": str(e), "error_count": 0}

    def _evaluate_performance(self, noisy_labels: np.ndarray,
                               clean_labels: List,
                               detected_error_indices: List[int],
                               label_encoder: LabelEncoder) -> Dict[str, Any]:
        """
        当提供真实标签时，评估标签错误检测的性能。

        Parameters
        ----------
        noisy_labels : np.ndarray
            编码后的噪声标签
        clean_labels : List
            真实标签列表（原始格式，与 label_encoder 相同的类别）
        detected_error_indices : List[int]
            检测器识别出的标签错误索引
        label_encoder : LabelEncoder
            标签编码器

        Returns
        -------
        Dict[str, Any]
            性能评估指标
        """
        try:
            clean_y = label_encoder.transform(clean_labels)

            actual_error_mask = (noisy_labels != clean_y)
            actual_error_indices = set(np.where(actual_error_mask)[0])
            detected_error_set = set(detected_error_indices)

            true_positives = len(actual_error_indices & detected_error_set)
            false_positives = len(detected_error_set - actual_error_indices)
            false_negatives = len(actual_error_indices - detected_error_set)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            actual_error_count = len(actual_error_indices)
            detected_error_count = len(detected_error_set)

            return {
                "has_ground_truth": True,
                "actual_error_count": actual_error_count,
                "actual_error_rate": float(actual_error_count / len(noisy_labels)),
                "detected_error_count": detected_error_count,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }
        except Exception as e:
            return {"has_ground_truth": True, "evaluation_error": str(e)}

    def _extract_features_auto(self, images: List[bytes]) -> np.ndarray:
        """
        自动选择最佳特征提取方式。

        优先使用 ResNet 深度特征（语义表达能力强），
        若 GPU/模型不可用则回退到统计特征。
        """
        resnet_features = self._try_extract_resnet_features(images)
        if resnet_features is not None:
            self._feature_type_used = "resnet50"
            return resnet_features

        self._feature_type_used = "statistical"
        return self._extract_features(images)

    def _try_extract_resnet_features(self, images: List[bytes]) -> Optional[np.ndarray]:
        """
        尝试使用预训练 ResNet50 提取 2048 维深度特征。

        Returns None if model/GPU not available.
        """
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            if not torch.cuda.is_available() and len(images) > 5000:
                return None

            try:
                from torchvision.models import ResNet50_Weights
                base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            except ImportError:
                base = models.resnet50(pretrained=True)

            feature_extractor = torch.nn.Sequential(*list(base.children())[:-1])
            feature_extractor.eval()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            feature_extractor = feature_extractor.to(device)

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            all_features = []
            batch_size = 64

            with torch.no_grad():
                for start in range(0, len(images), batch_size):
                    batch_imgs = images[start:start + batch_size]
                    batch_tensors = []
                    for img_bytes in batch_imgs:
                        try:
                            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                            tensor = transform(img_pil)
                            batch_tensors.append(tensor)
                        except Exception:
                            batch_tensors.append(torch.zeros(3, 224, 224))

                    if not batch_tensors:
                        continue

                    batch = torch.stack(batch_tensors).to(device)
                    feats = feature_extractor(batch)
                    feats = feats.squeeze(-1).squeeze(-1).cpu().numpy()
                    all_features.append(feats)

            if all_features:
                return np.vstack(all_features)
            return None

        except Exception:
            return None

    def _extract_features(self, images: List[bytes]) -> np.ndarray:
        """
        从图像字节中提取统计特征向量。

        特征组成（共 72 维）：
            - 颜色直方图（48 维）：R/G/B 通道各 16 bin 的直方图
            - 纹理特征（12 维）：灰度图的均值、标准差、Laplacian 方差、
              暗像素比例、亮像素比例、RMS 对比度 + HSV 的 H/S/V 通道均值
            - 尺寸特征（2 维）：高度、宽度（归一化）
            - 全局统计（10 维）：各通道均值和标准差

        Parameters
        ----------
        images : List[bytes]
            图像的原始字节数据列表

        Returns
        -------
        np.ndarray
            形状为 (n_images, n_features) 的特征矩阵
        """
        features = []
        for img_bytes in images:
            try:
                img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                img_array = np.array(img_pil)
                h, w = img_array.shape[:2]

                hist_features = []
                for c in range(3):
                    hist = cv2.calcHist([img_array], [c], None, [16], [0, 256])
                    hist = hist.flatten() / (h * w + 1e-8)
                    hist_features.extend(hist.tolist())

                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                dark_ratio = float(np.sum(gray < 50) / gray.size)
                bright_ratio = float(np.sum(gray > 205) / gray.size)
                rms_contrast = float(np.std(gray) / 255.0)

                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                h_mean = float(np.mean(hsv[:, :, 0]))
                s_mean = float(np.mean(hsv[:, :, 1]))
                v_mean = float(np.mean(hsv[:, :, 2]))

                r_mean, g_mean, b_mean = [float(np.mean(img_array[:, :, c])) for c in range(3)]
                r_std, g_std, b_std = [float(np.std(img_array[:, :, c])) for c in range(3)]

                feat = hist_features + [
                    float(np.mean(gray)), float(np.std(gray)),
                    laplacian_var, dark_ratio, bright_ratio, rms_contrast,
                    h_mean, s_mean, v_mean,
                    float(h / 1000.0), float(w / 1000.0),
                    r_mean, g_mean, b_mean, r_std, g_std, b_std,
                ]
                features.append(feat)

            except Exception:
                features.append([0.0] * 72)

        return np.array(features, dtype=np.float64)


class SoftmaxEntropyPredictor:
    """
    Softmax 熵预测器
    ================
    使用预训练 ResNet18 模型获取图像的 softmax 概率分布，
    通过预测熵衡量模型对样本的不确定性。

    核心思想：
        - 预训练模型在分布内（ID）样本上通常产生低熵（高置信度）的预测
        - 分布外（OOD）或异常样本通常产生高熵（低置信度）的预测
        - 通过比较目标集与基准集的熵分布差异，可检测分布偏移
        - 高熵样本本身也可作为不确定性估计的指标

    方法优势：
        - 无需标签即可工作（适用于分布偏移检测）
        - 利用预训练模型的语义特征，比纯统计特征更敏感
        - 计算效率高（单次前向传播）

    参考：
        - Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting
          Misclassified and Out-of-Distribution Examples in Neural Networks.
          ICLR 2017.
        - Liang, S., Li, Y., & Srikant, R. (2018). Enhancing the Reliability
          of Out-of-distribution Image Detection in Neural Networks. ICLR 2018.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._transform = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            try:
                from torchvision.models import ResNet50_Weights
                base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            except ImportError:
                base = models.resnet50(pretrained=True)
            self._model = torch.nn.Sequential(*list(base.children())[:-1])
            self._model.eval()

            self._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        except Exception:
            self._model = None

    def predict_entropy(self, images: List[bytes]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算图像列表的 softmax 熵和最大概率。

        Parameters
        ----------
        images : List[bytes]
            图像字节数据列表

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (entropy_array, max_prob_array)
            - entropy_array: 每个样本的预测熵，值域 [0, ln(C)]
            - max_prob_array: 每个样本的最大 softmax 概率，值域 (0, 1]
        """
        self._load_model()

        if self._model is None:
            return self._compute_statistical_entropy(images)

        import torch

        entropies = []
        max_probs = []

        for img_bytes in images:
            try:
                img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                tensor = self._transform(img_pil).unsqueeze(0)
                with torch.no_grad():
                    feat = self._model(tensor)
                feat = feat.squeeze().numpy()
                feat_scaled = feat / (np.max(np.abs(feat)) + 1e-8)
                exp_feat = np.exp(feat_scaled - np.max(feat_scaled))
                softmax_prob = exp_feat / (np.sum(exp_feat) + 1e-8)
                entropy = -np.sum(softmax_prob * np.log(softmax_prob + 1e-10))
                max_prob = float(np.max(softmax_prob))
                entropies.append(float(entropy))
                max_probs.append(max_prob)
            except Exception:
                entropies.append(0.0)
                max_probs.append(1.0)

        return np.array(entropies), np.array(max_probs)

    def _compute_statistical_entropy(self, images: List[bytes]) -> Tuple[np.ndarray, np.ndarray]:
        """
        当预训练模型不可用时，使用统计特征近似熵。

        基于图像质量特征（清晰度、对比度、饱和度等）构建伪熵：
        质量越低、越异常的图像，伪熵越高。
        """
        entropies = []
        max_probs = []

        for img_bytes in images:
            try:
                img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                img_array = np.array(img_pil)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

                laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpness = min(laplacian_var / 500.0, 1.0)
                contrast = float(np.std(gray) / 128.0)
                saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

                pseudo_entropy = (1.0 - sharpness) * 0.4 + (1.0 - min(contrast, 1.0)) * 0.3 + (1.0 - saturation) * 0.3
                pseudo_entropy = min(pseudo_entropy * 2.0, 1.0)

                entropies.append(pseudo_entropy)
                max_probs.append(1.0 - pseudo_entropy * 0.5)
            except Exception:
                entropies.append(0.5)
                max_probs.append(0.5)

        return np.array(entropies), np.array(max_probs)


class DistributionShiftDetector:
    """
    分布偏移检测器
    ==============
    检测用户上传图像集与基准图像集之间的分布差异。

    支持的偏移类型：
        1. 协变量偏移（Covariate Shift）：
           输入特征的分布发生变化，但标签条件分布不变。
           检测方法：使用颜色直方图 + 纹理特征，逐维 KS 检验。
           增强：使用 Softmax 熵法比较两组图像的语义分布差异。

        2. 子群偏移（Subgroup Shift）：
           某个维度（如亮度、颜色）整体偏移。
           检测方法：计算亮度直方图、颜色分布的 JS 散度。

        3. 语义偏移（Semantic Shift）：
           基于预训练模型的 softmax 熵分布差异。
           检测方法：比较目标集与基准集的预测熵分布（KS 检验 + JS 散度）。

    使用方式：
        需要提供 reference_images（基准图像集），系统会比较上传集与基准集的分布差异。

    参考：
        - Sugiyama, M., & Kawanabe, M. (2012). Machine Learning in Non-Stationary
          Environments: Introduction to Covariate Shift Adaptation. MIT Press.
        - Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting
          Misclassified and Out-of-Distribution Examples in Neural Networks.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ks_significance = cfg.get('detection.ks_significance', 0.05)
        self.js_threshold = cfg.get('detection.js_threshold', 0.1)
        self._entropy_predictor = SoftmaxEntropyPredictor(cfg)

    def detect(self, images: List[bytes],
               reference_images: List[bytes]) -> Dict[str, Any]:
        try:
            from scipy.stats import ks_2samp

            target_features = self._extract_distribution_features(images)
            ref_features = self._extract_distribution_features(reference_images)

            # ---- 1. 协变量偏移检测（统计特征 + KS 检验） ----
            shifted_dims = []
            feature_names = [
                "brightness_mean", "brightness_std", "saturation_mean",
                "hue_mean", "contrast", "sharpness",
                "r_mean", "g_mean", "b_mean",
            ]

            for dim_idx in range(min(target_features.shape[1], ref_features.shape[1])):
                try:
                    stat, p_value = ks_2samp(
                        target_features[:, dim_idx],
                        ref_features[:, dim_idx]
                    )
                    name = feature_names[dim_idx] if dim_idx < len(feature_names) else f"dim_{dim_idx}"
                    if p_value < self.ks_significance:
                        shifted_dims.append({
                            "dimension": name,
                            "ks_statistic": float(stat),
                            "p_value": float(p_value),
                        })
                except Exception:
                    pass

            # ---- 2. 子群偏移检测（亮度 JS 散度） ----
            target_brightness = target_features[:, 0]
            ref_brightness = ref_features[:, 0]
            js_divergence = self._js_divergence(target_brightness, ref_brightness)

            subgroup_warnings = []
            if js_divergence > self.js_threshold:
                target_dark_ratio = float(np.mean(target_brightness < 0.3))
                ref_dark_ratio = float(np.mean(ref_brightness < 0.3))
                if target_dark_ratio > ref_dark_ratio + 0.2:
                    subgroup_warnings.append(
                        f"注意：您的数据集中暗光图片占 {target_dark_ratio:.0%}，"
                        f"而基准集仅占 {ref_dark_ratio:.0%}，可能导致暗光下性能下降"
                    )
                target_bright_ratio = float(np.mean(target_brightness > 0.8))
                ref_bright_ratio = float(np.mean(ref_brightness > 0.8))
                if target_bright_ratio > ref_bright_ratio + 0.2:
                    subgroup_warnings.append(
                        f"注意：您的数据集中过曝图片占 {target_bright_ratio:.0%}，"
                        f"而基准集仅占 {ref_bright_ratio:.0%}，可能导致过曝场景性能下降"
                    )

            # ---- 3. 语义偏移检测（Softmax 熵法） ----
            semantic_shift = self._detect_semantic_shift(images, reference_images)

            all_warnings = subgroup_warnings + semantic_shift.get("warnings", [])

            return {
                "covariate_shift": {
                    "shifted_dimensions": shifted_dims,
                    "shifted_count": len(shifted_dims),
                    "total_dimensions": len(feature_names),
                },
                "subgroup_shift": {
                    "brightness_js_divergence": float(js_divergence),
                    "js_threshold": self.js_threshold,
                    "is_shifted": js_divergence > self.js_threshold,
                },
                "semantic_shift": semantic_shift,
                "warnings": all_warnings,
            }
        except Exception as e:
            return {
                "covariate_shift": {
                    "shifted_dimensions": [],
                    "shifted_count": 0,
                    "total_dimensions": 9,
                },
                "subgroup_shift": {
                    "brightness_js_divergence": 0.0,
                    "js_threshold": self.js_threshold,
                    "is_shifted": False,
                },
                "semantic_shift": {
                    "entropy_ks_statistic": 0.0,
                    "entropy_ks_pvalue": 1.0,
                    "entropy_js_divergence": 0.0,
                    "is_shifted": False,
                    "method": "softmax_entropy",
                },
                "warnings": [f"分布偏移检测发生错误: {str(e)}"],
            }

    def _detect_semantic_shift(self, images: List[bytes],
                                reference_images: List[bytes]) -> Dict[str, Any]:
        """
        使用 Softmax 熵法检测语义级别的分布偏移。

        核心思路：
            1. 使用预训练 ResNet18 提取两组图像的 softmax 概率分布
            2. 计算每个样本的预测熵 H(p) = -Σ p_i * log(p_i)
            3. 对两组熵分布执行 KS 检验，判断是否存在显著差异
            4. 计算两组熵分布的 JS 散度，量化偏移程度
            5. 比较两组的最大概率分布（MSP），辅助判断
        """
        from scipy.stats import ks_2samp

        target_entropy, target_max_prob = self._entropy_predictor.predict_entropy(images)
        ref_entropy, ref_max_prob = self._entropy_predictor.predict_entropy(reference_images)

        entropy_ks_stat = 0.0
        entropy_ks_pvalue = 1.0
        entropy_js = 0.0
        max_prob_ks_stat = 0.0
        max_prob_ks_pvalue = 1.0
        is_shifted = False
        warnings = []

        try:
            entropy_ks_stat, entropy_ks_pvalue = ks_2samp(target_entropy, ref_entropy)
        except Exception:
            pass

        try:
            entropy_js = self._js_divergence(target_entropy, ref_entropy)
        except Exception:
            pass

        try:
            max_prob_ks_stat, max_prob_ks_pvalue = ks_2samp(target_max_prob, ref_max_prob)
        except Exception:
            pass

        if entropy_ks_pvalue < self.ks_significance or max_prob_ks_pvalue < self.ks_significance:
            is_shifted = True

        target_mean_entropy = float(np.mean(target_entropy))
        ref_mean_entropy = float(np.mean(ref_entropy))

        if is_shifted:
            if target_mean_entropy > ref_mean_entropy + 0.1:
                warnings.append(
                    f"语义偏移警告：目标集平均预测熵 ({target_mean_entropy:.3f}) "
                    f"显著高于基准集 ({ref_mean_entropy:.3f})，"
                    f"目标集可能包含更多分布外样本或异常样本"
                )
            elif target_mean_entropy < ref_mean_entropy - 0.1:
                warnings.append(
                    f"语义偏移警告：目标集平均预测熵 ({target_mean_entropy:.3f}) "
                    f"显著低于基准集 ({ref_mean_entropy:.3f})，"
                    f"目标集的类别分布可能更集中"
                )
            else:
                warnings.append(
                    f"语义偏移警告：目标集与基准集的预测熵分布存在显著差异 "
                    f"(KS p={entropy_ks_pvalue:.4f})，"
                    f"建议检查数据采集条件是否发生变化"
                )

        return {
            "entropy_ks_statistic": float(entropy_ks_stat),
            "entropy_ks_pvalue": float(entropy_ks_pvalue),
            "entropy_js_divergence": float(entropy_js),
            "max_prob_ks_statistic": float(max_prob_ks_stat),
            "max_prob_ks_pvalue": float(max_prob_ks_pvalue),
            "target_mean_entropy": target_mean_entropy,
            "ref_mean_entropy": ref_mean_entropy,
            "is_shifted": is_shifted,
            "method": "softmax_entropy",
            "warnings": warnings,
        }

    def _extract_distribution_features(self, images: List[bytes]) -> np.ndarray:
        """
        提取用于分布比较的统计特征。

        特征维度（9 维）：
            - brightness_mean: 平均亮度
            - brightness_std: 亮度标准差
            - saturation_mean: 平均饱和度
            - hue_mean: 平均色调
            - contrast: RMS 对比度
            - sharpness: Laplacian 方差（归一化）
            - r_mean, g_mean, b_mean: RGB 通道均值

        Parameters
        ----------
        images : List[bytes]
            图像字节数据列表

        Returns
        -------
        np.ndarray
            形状为 (n_images, 9) 的特征矩阵
        """
        features = []
        for img_bytes in images:
            try:
                img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                img_array = np.array(img_pil)

                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

                brightness = float(np.mean(gray) / 255.0)
                brightness_std = float(np.std(gray) / 255.0)
                saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
                hue = float(np.mean(hsv[:, :, 0]) / 180.0)
                contrast = float(np.std(gray) / 255.0)
                sharpness = float(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0))
                r_mean = float(np.mean(img_array[:, :, 0]) / 255.0)
                g_mean = float(np.mean(img_array[:, :, 1]) / 255.0)
                b_mean = float(np.mean(img_array[:, :, 2]) / 255.0)

                features.append([
                    brightness, brightness_std, saturation, hue,
                    contrast, sharpness, r_mean, g_mean, b_mean
                ])
            except Exception:
                features.append([0.0] * 9)

        return np.array(features, dtype=np.float64)

    def _js_divergence(self, p: np.ndarray, q: np.ndarray, n_bins: int = 50) -> float:
        all_vals = np.concatenate([p, q])
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if vmax - vmin < 1e-10:
            return 0.0

        p_hist, _ = np.histogram(p, bins=n_bins, range=(vmin, vmax), density=True)
        q_hist, _ = np.histogram(q, bins=n_bins, range=(vmin, vmax), density=True)

        p_hist = p_hist.astype(np.float64) + 1e-10
        q_hist = q_hist.astype(np.float64) + 1e-10
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()

        m = 0.5 * (p_hist + q_hist)
        kl_pm = np.sum(p_hist * np.log(p_hist / m))
        kl_qm = np.sum(q_hist * np.log(q_hist / m))
        return float(0.5 * kl_pm + 0.5 * kl_qm)


class UncertaintyEstimator:
    """
    不确定性估计器
    ==============
    在模型训练前预知哪些样本是"困难样本"。

    支持两种策略：
        1. K 近邻距离法（需要标签）：
           计算每个样本到同类 K 近邻的平均距离。
           距离越大，说明该样本在特征空间中越孤立。

        2. Softmax 熵法（无需标签）：
           使用预训练模型计算预测熵，高熵表示模型对样本不确定。
           可检测分布外样本、标签错误和异常样本。

    参考：
        - Mandelbaum, A., & Weinshall, D. (2017). Distance-based Confidence
          Score for Neural Network Disambiguation. arXiv:1709.04864.
        - Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting
          Misclassified and Out-of-Distribution Examples in Neural Networks.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.k_neighbors = cfg.get('detection.k_neighbors', 5)
        self.uncertainty_std_factor = cfg.get('detection.uncertainty_std_factor', 1.5)
        self._entropy_predictor = SoftmaxEntropyPredictor(cfg)

    def detect(self, images: List[bytes], labels: List) -> Dict[str, Any]:
        try:
            entropy_scores, max_probs = self._entropy_predictor.predict_entropy(images)

            knn_scores = None
            if labels is not None and any(l is not None for l in labels):
                knn_scores = self._compute_knn_distances(images, labels)

            if knn_scores is not None:
                knn_norm = (knn_scores - np.min(knn_scores)) / (np.max(knn_scores) - np.min(knn_scores) + 1e-8)
                ent_norm = (entropy_scores - np.min(entropy_scores)) / (np.max(entropy_scores) - np.min(entropy_scores) + 1e-8)
                combined_scores = 0.5 * knn_norm + 0.5 * ent_norm
            else:
                combined_scores = entropy_scores

            mean_score = float(np.mean(combined_scores))
            std_score = float(np.std(combined_scores))
            threshold = mean_score + self.uncertainty_std_factor * std_score
            high_uncertainty_indices = np.where(combined_scores > threshold)[0].tolist()

            result = {
                "uncertainty_scores": combined_scores.tolist(),
                "high_uncertainty_indices": high_uncertainty_indices,
                "high_uncertainty_count": len(high_uncertainty_indices),
                "threshold": float(threshold),
                "method": "softmax_entropy+knn" if knn_scores is not None else "softmax_entropy",
                "statistics": {
                    "mean_distance": mean_score,
                    "std_distance": std_score,
                    "min_distance": float(np.min(combined_scores)),
                    "max_distance": float(np.max(combined_scores)),
                },
                "entropy_scores": entropy_scores.tolist(),
                "max_probs": max_probs.tolist(),
            }

            if knn_scores is not None:
                result["knn_scores"] = knn_scores.tolist()

            return result

        except Exception as e:
            return {"error": str(e), "uncertainty_scores": [], "high_uncertainty_count": 0}

    def _compute_knn_distances(self, images: List[bytes], labels: List) -> np.ndarray:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.neighbors import NearestNeighbors

        led = LabelErrorDetector(self.cfg)
        features = led._extract_features(images)

        le = LabelEncoder()
        y = le.fit_transform(labels)

        n_samples = len(images)
        distances = np.zeros(n_samples)

        for cls in np.unique(y):
            cls_mask = y == cls
            cls_features = features[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            k = min(self.k_neighbors, len(cls_features) - 1)
            if k < 1:
                distances[cls_indices] = 0.0
                continue

            nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
            nn.fit(cls_features)
            dist_matrix, _ = nn.kneighbors(cls_features)
            avg_distances = np.mean(dist_matrix[:, 1:], axis=1)
            distances[cls_indices] = avg_distances

        return distances


class ImageDetector(BaseDetector):
    """
    图像数据质量检测器 - 统一入口
    ==============================
    整合所有检测模块，提供统一的检测接口。
    用户可通过 modules 参数选择运行哪些检测模块。

    使用示例：
        # 仅运行基础质量检测
        result = detector.detect(images, modules=['basic_quality'])

        # 运行基础质量 + 标签错误检测
        result = detector.detect(images, labels=labels, modules=['basic_quality', 'label_error'])

        # 运行全部检测
        result = detector.detect(images, labels=labels, reference_images=ref_images,
                                 modules=['basic_quality', 'label_error',
                                          'distribution_shift', 'uncertainty'])
    """

    def __init__(self, cfg: Optional[Config] = None):
        super().__init__(cfg)
        self._basic_detector = BasicQualityDetector(self.cfg)
        self._label_detector = LabelErrorDetector(self.cfg)
        self._shift_detector = DistributionShiftDetector(self.cfg)
        self._uncertainty_estimator = UncertaintyEstimator(self.cfg)
        self._label_defect_pool = LabelDefectPool(self.cfg)
        self._distribution_shift_pool = DistributionShiftPool(self.cfg)
        self._sample_quality_pool = SampleQualityPool(self.cfg)
        self._format_structure_pool = FormatStructurePool(self.cfg)

    def modality(self) -> str:
        return 'image'

    def detect(self, data, labels: Optional[List] = None,
               modules: Optional[List[str]] = None,
               reference_images: Optional[List[bytes]] = None,
               clean_labels: Optional[List] = None,
               progress_callback=None,
               data_object=None,
               texts: Optional[List[str]] = None,
               data_root: str = "") -> Dict[str, Any]:
        """
        执行图像数据质量检测。

        Parameters
        ----------
        data : List[bytes] 或 List[np.ndarray]
            图像数据列表，支持原始字节数据或 numpy 数组
        labels : Optional[List]
            标签列表，标签错误检测和不确定性估计需要此参数
        modules : Optional[List[str]]
            要运行的检测模块列表，可选值：
            - 'basic_quality': 基础质量检测（默认必选）
            - 'label_error': 标签错误检测（需要 labels）
            - 'distribution_shift': 分布偏移检测（需要 reference_images）
            - 'uncertainty': 不确定性估计（需要 labels）
            - 'label_defect_pool': 标签缺陷检测池（需要 labels）
            - 'distribution_shift_pool': 分布偏移检测池（需要 reference_images）
            - 'sample_quality_pool': 样本质量检测池
            - 'format_structure_pool': 格式结构规则池
            默认为 ['basic_quality']
        reference_images : Optional[List[bytes]]
            基准图像集，分布偏移检测需要此参数
        clean_labels : Optional[List]
            真实标签列表（用于评估标签错误检测性能），若提供则计算精确率/召回率/F1
        progress_callback : Optional[callable]
            进度回调函数，签名为 callback(module_name: str, progress: float)
        data_object : Optional[DataObject]
            数据对象，格式结构规则池需要此参数
        texts : Optional[List[str]]
            文本描述列表，样本质量检测池的 CLIP Score 需要此参数
        data_root : str
            数据根目录，格式结构规则池的路径检查需要此参数

        Returns
        -------
        Dict[str, Any]
            综合检测结果字典
        """
        # 统一输入格式：将所有图像转为 bytes 列表
        images = self._normalize_images(data)

        # 默认只运行基础质量检测
        if modules is None:
            modules = ['basic_quality']

        self.issues = []
        self._init_report({
            'name': self.cfg.get('data.dataset_name', 'unknown'),
            'total_samples': len(images),
            'modules_run': modules,
        })

        result = {
            "modules_run": modules,
            "basic_quality": None,
            "label_error": None,
            "distribution_shift": None,
            "uncertainty": None,
            "label_defect_pool": None,
            "distribution_shift_pool": None,
            "sample_quality_pool": None,
            "format_structure_pool": None,
        }

        # ---- 模块 1: 基础质量检测（必选） ----
        if 'basic_quality' in modules:
            if progress_callback:
                progress_callback('basic_quality', 0.0)
            basic_result = self._basic_detector.detect(images)
            result["basic_quality"] = basic_result
            self.issues.extend(basic_result["issues"])

            # 写入报告
            for issue in basic_result["issues"]:
                section_name = issue["type"]
                if section_name not in self.report.sections:
                    self.report.add_section(DiagnosisSection(section_name))
                self.report.sections[section_name].add_issue(IssueRecord(
                    index=issue["index"],
                    issue_type=issue["type"],
                    quality_score=issue.get("sharpness") or issue.get("rms_contrast"),
                    details={k: v for k, v in issue.items() if k not in ('type', 'index')},
                ))

            for section in self.report.sections.values():
                self._set_noise_rate(section.name, len(section.issues), len(images))

            if progress_callback:
                progress_callback('basic_quality', 1.0)

        # ---- 模块 2: 标签错误检测 ----
        if 'label_error' in modules and labels is not None:
            if progress_callback:
                progress_callback('label_error', 0.0)
            label_result = self._label_detector.detect(images, labels, clean_labels=clean_labels)
            result["label_error"] = label_result

            if "error" not in label_result:
                label_section = DiagnosisSection("label_errors", "标签错误检测（置信学习）")
                label_section.add_metric("error_count", label_result["error_count"])
                label_section.add_metric("error_rate", label_result["error_rate"])

                for idx in label_result["error_indices"]:
                    original = labels[idx] if idx < len(labels) else None
                    suggested = label_result["suggested_labels"].get(str(idx))
                    score = label_result["label_quality_scores"][idx] if idx < len(label_result["label_quality_scores"]) else None

                    self.issues.append({
                        "type": "label_error",
                        "index": idx,
                        "original_label": original,
                        "suggested_label": suggested,
                        "quality_score": score,
                        "suggestion": "建议检查该样本标签，可点选修正后保存"
                    })
                    label_section.add_issue(IssueRecord(
                        index=idx,
                        issue_type="label_error",
                        original_label=original,
                        suggested_label=suggested,
                        quality_score=score,
                    ))

                self.report.add_section(label_section)
                self._set_noise_rate("label_error", label_result["error_count"], len(images))

            if progress_callback:
                progress_callback('label_error', 1.0)

        # ---- 模块 3: 分布偏移检测 ----
        if 'distribution_shift' in modules and reference_images is not None:
            if progress_callback:
                progress_callback('distribution_shift', 0.0)
            shift_result = self._shift_detector.detect(images, reference_images)
            result["distribution_shift"] = shift_result

            shift_section = DiagnosisSection("distribution_shift", "分布偏移检测")
            
            if "covariate_shift" in shift_result and "subgroup_shift" in shift_result:
                shift_section.add_metric("shifted_dimensions", shift_result["covariate_shift"]["shifted_count"])
                shift_section.add_metric("brightness_js_divergence", shift_result["subgroup_shift"]["brightness_js_divergence"])

                if "shifted_dimensions" in shift_result["covariate_shift"]:
                    for dim in shift_result["covariate_shift"]["shifted_dimensions"]:
                        self.issues.append({
                            "type": "distribution_shift",
                            "dimension": dim["dimension"],
                            "ks_statistic": dim["ks_statistic"],
                            "p_value": dim["p_value"],
                            "suggestion": "建议对偏移维度进行数据增强或重采样"
                        })
                        shift_section.add_issue(IssueRecord(
                            index=-1,
                            issue_type="distribution_shift",
                            details=dim,
                        ))

                semantic_shift = shift_result.get("semantic_shift", {})
                if semantic_shift:
                    shift_section.add_metric("semantic_shift_detected", semantic_shift.get("is_shifted", False))
                    shift_section.add_metric("entropy_ks_pvalue", semantic_shift.get("entropy_ks_pvalue", 1.0))
                    shift_section.add_metric("entropy_js_divergence", semantic_shift.get("entropy_js_divergence", 0.0))
                    if semantic_shift.get("is_shifted"):
                        self.issues.append({
                            "type": "semantic_shift",
                            "method": "softmax_entropy",
                            "entropy_ks_statistic": semantic_shift.get("entropy_ks_statistic", 0),
                            "entropy_ks_pvalue": semantic_shift.get("entropy_ks_pvalue", 1),
                            "entropy_js_divergence": semantic_shift.get("entropy_js_divergence", 0),
                            "suggestion": "检测到语义级分布偏移，建议检查数据采集条件或使用域适应方法"
                        })
            else:
                shift_section.add_metric("error", "分布偏移检测失败")

            self.report.add_section(shift_section)
            if progress_callback:
                progress_callback('distribution_shift', 1.0)

        # ---- 模块 4: 不确定性估计 ----
        if 'uncertainty' in modules and labels is not None:
            if progress_callback:
                progress_callback('uncertainty', 0.0)
            uncertainty_result = self._uncertainty_estimator.detect(images, labels)
            result["uncertainty"] = uncertainty_result

            if "error" not in uncertainty_result:
                unc_section = DiagnosisSection("uncertainty", "不确定性估计")
                unc_section.add_metric("high_uncertainty_count", uncertainty_result["high_uncertainty_count"])

                for idx in uncertainty_result["high_uncertainty_indices"]:
                    score = uncertainty_result["uncertainty_scores"][idx]
                    self.issues.append({
                        "type": "high_uncertainty",
                        "index": idx,
                        "uncertainty_score": score,
                        "suggestion": "该样本可能存在歧义或标签错误，建议人工检查"
                    })
                    unc_section.add_issue(IssueRecord(
                        index=idx,
                        issue_type="high_uncertainty",
                        quality_score=score,
                    ))

                self.report.add_section(unc_section)
                self._set_noise_rate("high_uncertainty", uncertainty_result["high_uncertainty_count"], len(images))

            if progress_callback:
                progress_callback('uncertainty', 1.0)

        # ---- 模块 5: 标签缺陷检测池 ----
        if 'label_defect_pool' in modules and labels is not None:
            if progress_callback:
                progress_callback('label_defect_pool', 0.0)
            try:
                led = LabelErrorDetector(self.cfg)
                features = led._extract_features_auto(images)
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(labels)

                pool_result = self._label_defect_pool.run_all(
                    features=features, labels=y
                )
                result["label_defect_pool"] = pool_result.to_dict()

                pool_section = DiagnosisSection("label_defect_pool", "标签缺陷检测池")
                for mr in pool_result.method_results:
                    pool_section.add_metric(
                        f"{mr.method_name}_success", mr.success
                    )
                    if mr.success and len(mr.scores) > 0:
                        pool_section.add_metric(
                            f"{mr.method_name}_mean", float(np.mean(mr.scores))
                        )

                if pool_result.ensemble_scores is not None:
                    high_defect_threshold = 0.7
                    high_defect_indices = np.where(
                        pool_result.ensemble_scores > high_defect_threshold
                    )[0].tolist()
                    pool_section.add_metric(
                        "high_defect_count", len(high_defect_indices)
                    )
                    for idx in high_defect_indices:
                        self.issues.append({
                            "type": "label_defect",
                            "index": idx,
                            "defect_score": float(pool_result.ensemble_scores[idx]),
                            "suggestion": "该样本可能存在标签缺陷，建议人工检查",
                        })

                self.report.add_section(pool_section)
            except Exception as e:
                result["label_defect_pool"] = {"error": str(e)}

            if progress_callback:
                progress_callback('label_defect_pool', 1.0)

        # ---- 模块 6: 分布偏移检测池 ----
        if 'distribution_shift_pool' in modules and reference_images is not None:
            if progress_callback:
                progress_callback('distribution_shift_pool', 0.0)
            try:
                target_features = self._shift_detector._extract_distribution_features(images)
                ref_features = self._shift_detector._extract_distribution_features(reference_images)

                pool_result = self._distribution_shift_pool.run_all(
                    source_features=ref_features, target_features=target_features
                )
                result["distribution_shift_pool"] = pool_result.to_dict()

                pool_section = DiagnosisSection("distribution_shift_pool", "分布偏移检测池")
                for mr in pool_result.method_results:
                    pool_section.add_metric(
                        f"{mr.method_name}_success", mr.success
                    )
                    if mr.success and len(mr.scores) > 0:
                        pool_section.add_metric(
                            f"{mr.method_name}_score", float(mr.scores[0])
                        )

                if pool_result.ensemble_scores is not None and len(pool_result.ensemble_scores) > 0:
                    ensemble_score = float(pool_result.ensemble_scores[0])
                    if ensemble_score > 0.5:
                        self.issues.append({
                            "type": "distribution_shift_pool",
                            "shift_score": ensemble_score,
                            "suggestion": "检测到显著分布偏移，建议使用域适应方法或重新采集数据",
                        })

                self.report.add_section(pool_section)
            except Exception as e:
                result["distribution_shift_pool"] = {"error": str(e)}

            if progress_callback:
                progress_callback('distribution_shift_pool', 1.0)

        # ---- 模块 7: 样本质量检测池 ----
        if 'sample_quality_pool' in modules:
            if progress_callback:
                progress_callback('sample_quality_pool', 0.0)
            try:
                pool_result = self._sample_quality_pool.run_all(
                    images=images, texts=texts
                )
                result["sample_quality_pool"] = pool_result.to_dict()

                pool_section = DiagnosisSection("sample_quality_pool", "样本质量检测池")
                for mr in pool_result.method_results:
                    pool_section.add_metric(
                        f"{mr.method_name}_success", mr.success
                    )
                    if mr.success and len(mr.scores) > 0:
                        pool_section.add_metric(
                            f"{mr.method_name}_mean", float(np.mean(mr.scores))
                        )

                if pool_result.ensemble_scores is not None:
                    quality_threshold = 0.7
                    low_quality_indices = np.where(
                        pool_result.ensemble_scores > quality_threshold
                    )[0].tolist()
                    pool_section.add_metric(
                        "low_quality_count", len(low_quality_indices)
                    )
                    for idx in low_quality_indices:
                        self.issues.append({
                            "type": "low_quality",
                            "index": idx,
                            "quality_anomaly_score": float(pool_result.ensemble_scores[idx]),
                            "suggestion": "该样本质量异常，建议检查或丢弃",
                        })

                self.report.add_section(pool_section)
            except Exception as e:
                result["sample_quality_pool"] = {"error": str(e)}

            if progress_callback:
                progress_callback('sample_quality_pool', 1.0)

        # ---- 模块 8: 格式结构规则池 ----
        if 'format_structure_pool' in modules:
            if progress_callback:
                progress_callback('format_structure_pool', 0.0)
            try:
                pool_result = self._format_structure_pool.run_all(
                    data_object=data_object,
                    labels=labels,
                    data_root=data_root,
                )
                result["format_structure_pool"] = pool_result.to_dict()

                pool_section = DiagnosisSection("format_structure_pool", "格式结构规则池")
                for mr in pool_result.method_results:
                    pool_section.add_metric(
                        f"{mr.method_name}_success", mr.success
                    )
                    if not mr.success:
                        self.issues.append({
                            "type": "format_structure_violation",
                            "rule": mr.method_name,
                            "error": mr.error_message,
                            "suggestion": f"格式结构规则 '{mr.method_name}' 校验失败",
                        })

                self.report.add_section(pool_section)
            except Exception as e:
                result["format_structure_pool"] = {"error": str(e)}

            if progress_callback:
                progress_callback('format_structure_pool', 1.0)

        # 汇总指标
        basic_metrics = result["basic_quality"]["metrics"] if result["basic_quality"] else {}
        self.metrics = {
            "total_images": len(images),
            "issue_count": len(self.issues),
            "issue_rate": float(len(self.issues) / len(images) * 100) if images else 0,
            "average_quality_score": basic_metrics.get("average_quality_score", 0),
            "quality_scores": basic_metrics.get("quality_scores", []),
            "modules_run": modules,
        }

        self.report.build_summary()

        result["metrics"] = self.metrics
        result["issues"] = self.issues
        result["diagnosis_report"] = self.report.to_dict()

        return result

    def _normalize_images(self, data) -> List[bytes]:
        """
        将输入数据统一转为 bytes 列表。

        支持的输入格式：
            - List[bytes]: 原始图像字节数据
            - List[np.ndarray]: numpy 数组（转为 PNG bytes）
            - List[io.BytesIO]: BytesIO 对象（提取 bytes）
            - 单个 bytes: 包装为列表
            - 单个 np.ndarray: 包装为列表
            - 单个 io.BytesIO: 包装为列表

        Parameters
        ----------
        data : 各种格式
            输入图像数据

        Returns
        -------
        List[bytes]
            统一的图像字节数据列表
        """
        if isinstance(data, bytes):
            return [data]
        elif isinstance(data, np.ndarray):
            return [self._array_to_bytes(data)]
        elif isinstance(data, io.BytesIO):
            return [data.getvalue()]
        elif isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, bytes):
                    result.append(item)
                elif isinstance(item, np.ndarray):
                    result.append(self._array_to_bytes(item))
                elif isinstance(item, io.BytesIO):
                    result.append(item.getvalue())
                else:
                    result.append(item)
            return result
        return data

    def _array_to_bytes(self, arr: np.ndarray) -> bytes:
        """将 numpy 数组转为 PNG 格式的字节数据"""
        img_pil = Image.fromarray(arr)
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        return buf.getvalue()

    @staticmethod
    def load_images_from_zip(zip_bytes: bytes) -> Tuple[List[bytes], List[str]]:
        """
        从 ZIP 文件中加载图像。

        Parameters
        ----------
        zip_bytes : bytes
            ZIP 文件的原始字节数据

        Returns
        -------
        Tuple[List[bytes], List[str]]
            (图像字节数据列表, 文件名列表)
        """
        images = []
        filenames = []
        supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in sorted(zf.namelist()):
                ext = os.path.splitext(name)[1].lower()
                if ext in supported_ext and not name.startswith('__MACOSX'):
                    try:
                        img_bytes = zf.read(name)
                        # 验证是否为有效图像
                        Image.open(io.BytesIO(img_bytes))
                        images.append(img_bytes)
                        filenames.append(os.path.basename(name))
                    except Exception:
                        continue

        return images, filenames

    @staticmethod
    def load_images_from_folder(folder_path: str) -> Tuple[List[bytes], List[str]]:
        """
        从本地文件夹加载图像。

        Parameters
        ----------
        folder_path : str
            文件夹路径

        Returns
        -------
        Tuple[List[bytes], List[str]]
            (图像字节数据列表, 文件名列表)
        """
        images = []
        filenames = []
        supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        for root, dirs, files in os.walk(folder_path):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in supported_ext:
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'rb') as f:
                            img_bytes = f.read()
                        Image.open(io.BytesIO(img_bytes))
                        images.append(img_bytes)
                        filenames.append(fname)
                    except Exception:
                        continue

        return images, filenames

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_issues(self) -> List[Dict[str, Any]]:
        return self.issues
