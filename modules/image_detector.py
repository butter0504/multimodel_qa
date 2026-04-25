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
        1. 使用预训练 ResNet18 提取图像特征向量
        2. 基于特征向量训练 RandomForest 分类器
        3. 使用交叉验证获取样本的预测概率分布
        4. 调用 cleanlab 的 get_label_quality_scores 计算标签质量分数
        5. 质量分数低于阈值的样本被标记为潜在标签错误
        6. 分类器的预测结果作为建议修正标签

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
        # 标签质量分数阈值：低于此值的样本被标记为潜在标签错误
        self.quality_threshold = cfg.get('detection.threshold', 0.6)
        # 交叉验证折数
        self.cv_folds = cfg.get('detection.cross_validation_folds', 5)
        # 随机森林分类器参数
        self.n_estimators = cfg.get('detection.n_estimators', 200)
        self.random_state = cfg.get('detection.random_state', 42)

    def detect(self, images: List[bytes], labels: List) -> Dict[str, Any]:
        """
        使用置信学习检测图像标签错误。

        Parameters
        ----------
        images : List[bytes]
            图像的原始字节数据列表
        labels : List
            对应的标签列表，支持任意类型的标签

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
            - debug_info: 调试信息（类别分布、分数统计等）
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_predict
            from sklearn.preprocessing import LabelEncoder
            from cleanlab.rank import get_label_quality_scores

            # Step 1: 提取图像统计特征
            # 使用颜色直方图 + 纹理特征作为轻量级特征表示
            # 这种方式无需 GPU，适合快速检测
            features = self._extract_features(images)

            # Step 2: 标签编码
            # 将任意类型的标签（字符串、数字等）统一转为 0, 1, 2, ... 的整数编码
            le = LabelEncoder()
            y = le.fit_transform(labels)
            n_classes = len(le.classes_)

            # 交叉验证要求每类至少有 cv_folds 个样本
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

            # Step 3: 交叉验证获取预测概率
            # 使用 RandomForest 作为分类器，通过交叉验证避免过拟合
            # cross_val_predict 返回每个样本在"未参与训练"的模型上的预测概率
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            pred_probs = cross_val_predict(
                model, features, y, cv=actual_cv_folds, method='predict_proba'
            )

            # Step 4: 计算标签质量分数
            # cleanlab 的 get_label_quality_scores 基于预测概率和真实标签，
            # 使用自适应阈值计算每个样本的标签质量分数
            # 分数越低，标签越可能有问题
            label_quality_scores = get_label_quality_scores(y, pred_probs)

            # Step 5: 识别标签错误
            # 质量分数低于阈值的样本被标记为潜在标签错误
            error_indices = np.where(label_quality_scores < self.quality_threshold)[0].tolist()

            # Step 6: 生成建议修正标签
            # 使用全量数据训练的模型预测作为建议标签
            model.fit(features, y)
            predictions = model.predict(features)
            suggested_labels = {}
            for idx in error_indices:
                suggested_labels[str(idx)] = int(predictions[idx])

            # Step 7: 计算置信学习的噪声矩阵
            # 噪声矩阵 T[i][j] 表示真实标签为 i 但被标注为 j 的概率
            # 对角线元素越大，标签越可靠
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

            return {
                "error_count": len(error_indices),
                "error_rate": len(error_indices) / len(y) if len(y) > 0 else 0,
                "error_indices": error_indices,
                "suggested_labels": suggested_labels,
                "label_quality_scores": label_quality_scores.tolist(),
                "confidence_thresholds": thresholds,
                "confident_joint_matrix": C_confident.tolist(),
                "noise_matrix": noise_matrix.tolist(),
                "class_names": le.classes_.tolist(),
                "debug_info": {
                    "n_classes": n_classes,
                    "class_counts": class_counts.tolist(),
                    "cv_folds_used": actual_cv_folds,
                    "mean_quality_score": float(np.mean(label_quality_scores)),
                    "min_quality_score": float(np.min(label_quality_scores)),
                    "max_quality_score": float(np.max(label_quality_scores)),
                },
            }

        except ImportError as e:
            return {"error": f"缺少依赖库: {str(e)}", "error_count": 0}
        except Exception as e:
            return {"error": str(e), "error_count": 0}

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

                # 颜色直方图特征（每通道 16 bin，共 48 维）
                hist_features = []
                for c in range(3):
                    hist = cv2.calcHist([img_array], [c], None, [16], [0, 256])
                    hist = hist.flatten() / (h * w + 1e-8)
                    hist_features.extend(hist.tolist())

                # 灰度图统计特征
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                dark_ratio = float(np.sum(gray < 50) / gray.size)
                bright_ratio = float(np.sum(gray > 205) / gray.size)
                rms_contrast = float(np.std(gray) / 255.0)

                # HSV 颜色空间特征
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                h_mean = float(np.mean(hsv[:, :, 0]))
                s_mean = float(np.mean(hsv[:, :, 1]))
                v_mean = float(np.mean(hsv[:, :, 2]))

                # 通道统计特征
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


class DistributionShiftDetector:
    """
    分布偏移检测器
    ==============
    检测用户上传图像集与基准图像集之间的分布差异。

    支持的偏移类型：
        1. 协变量偏移（Covariate Shift）：
           输入特征的分布发生变化，但标签条件分布不变。
           检测方法：使用颜色直方图 + 纹理特征，逐维 KS 检验。

        2. 子群偏移（Subgroup Shift）：
           某个维度（如亮度、颜色）整体偏移。
           检测方法：计算亮度直方图、颜色分布的 JS 散度。

    使用方式：
        需要提供 reference_images（基准图像集），系统会比较上传集与基准集的分布差异。

    参考：
        - Sugiyama, M., & Kawanabe, M. (2012). Machine Learning in Non-Stationary
          Environments: Introduction to Covariate Shift Adaptation. MIT Press.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # KS 检验的显著性水平，p 值低于此值认为该维度存在分布偏移
        self.ks_significance = cfg.get('detection.ks_significance', 0.05)
        # JS 散度阈值，超过此值认为存在子群偏移
        self.js_threshold = cfg.get('detection.js_threshold', 0.1)

    def detect(self, images: List[bytes],
               reference_images: List[bytes]) -> Dict[str, Any]:
        """
        检测上传图像集与基准图像集之间的分布偏移。

        Parameters
        ----------
        images : List[bytes]
            用户上传的图像集
        reference_images : List[bytes]
            基准图像集（如训练集的样本）

        Returns
        -------
        Dict[str, Any]
            包含以下键的字典：
            - covariate_shift: 协变量偏移检测结果
            - subgroup_shift: 子群偏移检测结果
            - warnings: 预警信息列表
        """
        from scipy.stats import ks_2samp

        # 提取两组图像的特征
        target_features = self._extract_distribution_features(images)
        ref_features = self._extract_distribution_features(reference_images)

        # ---- 1. 协变量偏移检测 ----
        # 对每个特征维度执行 Kolmogorov-Smirnov 双样本检验
        # KS 检验是比较两个经验累积分布函数（ECDF）的最大差异
        # H0: 两个样本来自同一分布
        # p < significance → 拒绝 H0 → 该维度存在分布偏移
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

        # ---- 2. 子群偏移检测 ----
        # 计算亮度直方图的 Jensen-Shannon 散度
        # JS 散度是 KL 散度的对称化版本，值域 [0, 1]
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
            "warnings": subgroup_warnings,
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
        """
        计算两个连续分布的 Jensen-Shannon 散度。

        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)，其中 M = 0.5*(P+Q)
        JS 散度是 KL 散度的对称化、有界版本，值域 [0, ln(2)]

        Parameters
        ----------
        p, q : np.ndarray
            两个样本的一维特征值数组
        n_bins : int
            直方图 bin 数量

        Returns
        -------
        float
            JS 散度值
        """
        p_hist, _ = np.histogram(p, bins=n_bins, range=(0, 1), density=True)
        q_hist, _ = np.histogram(q, bins=n_bins, range=(0, 1), density=True)

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

    策略：特征空间密度估计
        计算每个样本到同类 K 近邻的平均距离。
        距离越大，说明该样本在特征空间中越孤立，越可能是：
        - 标签错误（特征与同类不一致）
        - 边界样本（位于类别边界附近）
        - 异常样本（数据采集或标注出错）

    这种方法不需要训练模型，计算速度快，适合作为预处理步骤。

    参考：
        - Mandelbaum, A., & Weinshall, D. (2017). Distance-based Confidence
          Score for Neural Network Disambiguation. arXiv:1709.04864.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # K 近邻的 K 值
        self.k_neighbors = cfg.get('detection.k_neighbors', 5)
        # 不确定性阈值：距离超过均值 + 此值×标准差 的样本标记为高不确定性
        self.uncertainty_std_factor = cfg.get('detection.uncertainty_std_factor', 1.5)

    def detect(self, images: List[bytes], labels: List) -> Dict[str, Any]:
        """
        估计图像样本的不确定性。

        Parameters
        ----------
        images : List[bytes]
            图像字节数据列表
        labels : List
            对应的标签列表

        Returns
        -------
        Dict[str, Any]
            包含以下键的字典：
            - uncertainty_scores: 每个样本的不确定性分数
            - high_uncertainty_indices: 高不确定性样本索引
            - high_uncertainty_count: 高不确定性样本数量
            - statistics: 不确定性分数统计信息
        """
        try:
            from sklearn.preprocessing import LabelEncoder
            from sklearn.neighbors import NearestNeighbors

            # 提取特征（复用 LabelErrorDetector 的特征提取方法）
            led = LabelErrorDetector(self.cfg)
            features = led._extract_features(images)

            le = LabelEncoder()
            y = le.fit_transform(labels)

            # 对每个类别，计算样本到同类 K 近邻的平均距离
            n_samples = len(images)
            distances = np.zeros(n_samples)

            for cls in np.unique(y):
                cls_mask = y == cls
                cls_features = features[cls_mask]
                cls_indices = np.where(cls_mask)[0]

                # 如果该类样本数不足 K+1，使用所有样本
                k = min(self.k_neighbors, len(cls_features) - 1)
                if k < 1:
                    distances[cls_indices] = 0.0
                    continue

                # 计算 K 近邻距离
                nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
                nn.fit(cls_features)
                dist_matrix, _ = nn.kneighbors(cls_features)
                # 排除自身（距离为 0 的第一个邻居）
                avg_distances = np.mean(dist_matrix[:, 1:], axis=1)
                distances[cls_indices] = avg_distances

            # 识别高不确定性样本
            mean_dist = float(np.mean(distances))
            std_dist = float(np.std(distances))
            threshold = mean_dist + self.uncertainty_std_factor * std_dist
            high_uncertainty_indices = np.where(distances > threshold)[0].tolist()

            return {
                "uncertainty_scores": distances.tolist(),
                "high_uncertainty_indices": high_uncertainty_indices,
                "high_uncertainty_count": len(high_uncertainty_indices),
                "threshold": float(threshold),
                "statistics": {
                    "mean_distance": mean_dist,
                    "std_distance": std_dist,
                    "min_distance": float(np.min(distances)),
                    "max_distance": float(np.max(distances)),
                },
            }

        except Exception as e:
            return {"error": str(e), "uncertainty_scores": [], "high_uncertainty_count": 0}


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

    def modality(self) -> str:
        return 'image'

    def detect(self, data, labels: Optional[List] = None,
               modules: Optional[List[str]] = None,
               reference_images: Optional[List[bytes]] = None,
               progress_callback=None) -> Dict[str, Any]:
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
            默认为 ['basic_quality']
        reference_images : Optional[List[bytes]]
            基准图像集，分布偏移检测需要此参数
        progress_callback : Optional[callable]
            进度回调函数，签名为 callback(module_name: str, progress: float)

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
            label_result = self._label_detector.detect(images, labels)
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
            shift_section.add_metric("shifted_dimensions", shift_result["covariate_shift"]["shifted_count"])
            shift_section.add_metric("brightness_js_divergence", shift_result["subgroup_shift"]["brightness_js_divergence"])

            for dim in shift_result["covariate_shift"]["shifted_dims"]:
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
            - 单个 bytes: 包装为列表
            - 单个 np.ndarray: 包装为列表

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
        elif isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, bytes):
                    result.append(item)
                elif isinstance(item, np.ndarray):
                    result.append(self._array_to_bytes(item))
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
