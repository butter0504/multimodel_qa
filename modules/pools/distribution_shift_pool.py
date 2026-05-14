from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional

from .base_pool import BasePool
from ..config import Config


class DistributionShiftPool(BasePool):
    """
    分布偏移检测池
    ==============
    包含4种分布偏移检测方法，输出归一化后的偏移程度分数。

    方法列表：
        1. ks_test              - 两样本 Kolmogorov-Smirnov 检验
        2. mmd_test             - 最大均值差异，使用 RBF 核
        3. wasserstein_distance - Earth Mover's Distance
        4. domain_classifier_auc - 训练 LightGBM 分类器，输出 AUC
    """

    def pool_name(self) -> str:
        return "distribution_shift"

    def _register_methods(self):
        self._methods = {
            "ks_test": self.ks_test,
            "mmd_test": self.mmd_test,
            "wasserstein_distance": self.wasserstein_distance,
            "domain_classifier_auc": self.domain_classifier_auc,
        }

    def ks_test(self, source_features: np.ndarray,
                target_features: np.ndarray, **kwargs) -> np.ndarray:
        """
        两样本 Kolmogorov-Smirnov 检验。

        逐特征维度对源域和目标域执行 KS 检验，
        取各维度 KS 统计量的均值作为整体偏移分数。

        Parameters
        ----------
        source_features : np.ndarray
            源域特征矩阵 (n_source, n_features)
        target_features : np.ndarray
            目标域特征矩阵 (n_target, n_features)

        Returns
        -------
        np.ndarray
            偏移程度分数，形状 (1,)，值域 [0, 1]
        """
        from scipy.stats import ks_2samp

        n_dims = min(source_features.shape[1], target_features.shape[1])
        ks_stats = []
        p_values = []

        for dim in range(n_dims):
            try:
                stat, p_val = ks_2samp(source_features[:, dim], target_features[:, dim])
                ks_stats.append(stat)
                p_values.append(p_val)
            except Exception:
                continue

        if not ks_stats:
            return np.array([0.0], dtype=np.float64)

        mean_ks = float(np.mean(ks_stats))
        shift_ratio = float(np.mean(np.array(p_values) < 0.05))
        shift_score = 0.5 * mean_ks + 0.5 * shift_ratio
        return np.array([np.clip(shift_score, 0.0, 1.0)], dtype=np.float64)

    def mmd_test(self, source_features: np.ndarray,
                 target_features: np.ndarray, **kwargs) -> np.ndarray:
        """
        最大均值差异（Maximum Mean Discrepancy），使用 RBF 核。

        MMD 衡量两个分布在再生核希尔伯特空间中的距离。
        使用 RBF 核的 MMD^2 无偏估计。

        Parameters
        ----------
        source_features : np.ndarray
            源域特征矩阵 (n_source, n_features)
        target_features : np.ndarray
            目标域特征矩阵 (n_target, n_features)

        Returns
        -------
        np.ndarray
            偏移程度分数，形状 (1,)，值域 [0, 1]
        """
        from sklearn.metrics.pairwise import rbf_kernel

        X = source_features
        Y = target_features

        pairwise_dist = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
        median_dist = np.median(pairwise_dist)
        gamma = 1.0 / (2 * median_dist ** 2 + 1e-8)

        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)

        n = len(X)
        m = len(Y)

        mmd_XX = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) if n > 1 else 0.0
        mmd_YY = (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) if m > 1 else 0.0
        mmd_XY = K_XY.sum() / (n * m)

        mmd_sq = mmd_XX + mmd_YY - 2 * mmd_XY
        mmd_value = np.sqrt(max(mmd_sq, 0.0))

        normalized = min(mmd_value / (mmd_value + 1.0), 1.0)
        return np.array([float(normalized)], dtype=np.float64)

    def wasserstein_distance(self, source_features: np.ndarray,
                             target_features: np.ndarray, **kwargs) -> np.ndarray:
        """
        Earth Mover's Distance（Wasserstein 距离）。

        逐特征维度计算一维 Wasserstein 距离，
        取各维度距离的均值作为整体偏移分数。

        Parameters
        ----------
        source_features : np.ndarray
            源域特征矩阵 (n_source, n_features)
        target_features : np.ndarray
            目标域特征矩阵 (n_target, n_features)

        Returns
        -------
        np.ndarray
            偏移程度分数，形状 (1,)，值域 [0, 1]
        """
        from scipy.stats import wasserstein_distance as wd

        n_dims = min(source_features.shape[1], target_features.shape[1])
        distances = []

        for dim in range(n_dims):
            try:
                d = wd(source_features[:, dim], target_features[:, dim])
                distances.append(d)
            except Exception:
                continue

        if not distances:
            return np.array([0.0], dtype=np.float64)

        mean_dist = float(np.mean(distances))
        normalized = mean_dist / (mean_dist + 1.0)
        return np.array([float(normalized)], dtype=np.float64)

    def domain_classifier_auc(self, source_features: np.ndarray,
                              target_features: np.ndarray, **kwargs) -> np.ndarray:
        """
        训练 LightGBM 域分类器，输出 AUC 作为偏移分数。

        将源域标记为0，目标域标记为1，训练域分类器。
        AUC 越接近1，说明两个域越容易区分，偏移越大。
        AUC=0.5 表示无法区分，即无偏移。

        Parameters
        ----------
        source_features : np.ndarray
            源域特征矩阵 (n_source, n_features)
        target_features : np.ndarray
            目标域特征矩阵 (n_target, n_features)

        Returns
        -------
        np.ndarray
            偏移程度分数，形状 (1,)，值域 [0, 1]
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_predict

        n_source = len(source_features)
        n_target = len(target_features)

        X = np.vstack([source_features, target_features])
        y = np.concatenate([np.zeros(n_source), np.ones(n_target)])

        try:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.cfg.get('detection.random_state', 42),
                verbose=-1,
                n_jobs=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.cfg.get('detection.random_state', 42),
            )

        try:
            min_class = min(n_source, n_target)
            cv_folds = min(5, min_class)
            if cv_folds >= 2:
                pred_probs = cross_val_predict(model, X, y, cv=cv_folds, method='predict_proba')[:, 1]
            else:
                model.fit(X, y)
                pred_probs = model.predict_proba(X)[:, 1]

            auc = roc_auc_score(y, pred_probs)
        except Exception:
            model.fit(X, y)
            pred_probs = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, pred_probs)

        shift_score = 2.0 * abs(auc - 0.5)
        return np.array([float(np.clip(shift_score, 0.0, 1.0))], dtype=np.float64)
