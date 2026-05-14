from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional

from .base_pool import BasePool, MethodResult
from ..config import Config


class LabelDefectPool(BasePool):
    """
    标签缺陷检测池
    ==============
    包含5种标签缺陷检测方法，每种方法输出缺陷置信度分数（0~1）。

    方法列表：
        1. cleanlab_detection  - 调用 cleanlab 置信学习框架
        2. knn_consistency     - K近邻标签一致性（k=15）
        3. isolation_forest    - 特征+标签拼接后做异常检测
        4. ensemble_divergence - SVM+RF+MLP 三模型预测熵
        5. loss_trajectory     - 训练 ResNet-18，取交叉熵损失
    """

    def pool_name(self) -> str:
        return "label_defect"

    def _register_methods(self):
        self._methods = {
            "cleanlab_detection": self.cleanlab_detection,
            "knn_consistency": self.knn_consistency,
            "isolation_forest": self.isolation_forest,
            "ensemble_divergence": self.ensemble_divergence,
            "loss_trajectory": self.loss_trajectory,
        }

    def cleanlab_detection(self, features: np.ndarray, labels: np.ndarray,
                           **kwargs) -> np.ndarray:
        """
        调用 cleanlab 置信学习框架检测标签缺陷。

        使用交叉验证获取预测概率，然后调用 cleanlab 的
        get_label_quality_scores 计算每个样本的标签质量分数，
        反转后作为缺陷置信度。

        Parameters
        ----------
        features : np.ndarray
            特征矩阵 (n_samples, n_features)
        labels : np.ndarray
            标签数组 (n_samples,)

        Returns
        -------
        np.ndarray
            缺陷置信度分数 (n_samples,)，值域 [0, 1]
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import LabelEncoder
        from cleanlab.rank import get_label_quality_scores

        le = LabelEncoder()
        y = le.fit_transform(labels)
        n_classes = len(le.classes_)

        class_counts = np.bincount(y)
        min_count = int(np.min(class_counts))
        cv_folds = min(self.cfg.get('detection.cross_validation_folds', 5), min_count)
        if cv_folds < 2:
            return np.zeros(len(y), dtype=np.float64)

        model = RandomForestClassifier(
            n_estimators=self.cfg.get('detection.n_estimators', 200),
            random_state=self.cfg.get('detection.random_state', 42),
            n_jobs=-1,
        )
        pred_probs = cross_val_predict(
            model, features, y, cv=cv_folds, method='predict_proba'
        )

        quality_scores = get_label_quality_scores(y, pred_probs)
        defect_scores = 1.0 - quality_scores
        defect_scores = (defect_scores - defect_scores.min()) / (defect_scores.max() - defect_scores.min() + 1e-8)
        return defect_scores.astype(np.float64)

    def knn_consistency(self, features: np.ndarray, labels: np.ndarray,
                        k: int = 15, **kwargs) -> np.ndarray:
        """
        计算K近邻标签一致性。

        对每个样本找到特征空间中最近的k个邻居，
        计算邻居中与该样本标签不一致的比例作为缺陷置信度。

        Parameters
        ----------
        features : np.ndarray
            特征矩阵 (n_samples, n_features)
        labels : np.ndarray
            标签数组 (n_samples,)
        k : int
            近邻数，默认15

        Returns
        -------
        np.ndarray
            缺陷置信度分数 (n_samples,)，值域 [0, 1]
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(labels)

        actual_k = min(k, len(features) - 1)
        if actual_k < 1:
            return np.zeros(len(y), dtype=np.float64)

        nn = NearestNeighbors(n_neighbors=actual_k + 1, metric='euclidean')
        nn.fit(features)
        _, indices = nn.kneighbors(features)

        neighbor_labels = y[indices[:, 1:]]
        inconsistency = np.mean(neighbor_labels != y[:, np.newaxis], axis=1)
        return inconsistency.astype(np.float64)

    def isolation_forest(self, features: np.ndarray, labels: np.ndarray,
                         **kwargs) -> np.ndarray:
        """
        特征+标签拼接后做异常检测。

        将标签编码后与特征拼接，使用 Isolation Forest 检测异常样本。
        异常样本的标签可能存在缺陷。

        Parameters
        ----------
        features : np.ndarray
            特征矩阵 (n_samples, n_features)
        labels : np.ndarray
            标签数组 (n_samples,)

        Returns
        -------
        np.ndarray
            缺陷置信度分数 (n_samples,)，值域 [0, 1]
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        le = LabelEncoder()
        y = le.fit_transform(labels)
        n_classes = len(le.classes_)

        ohe = OneHotEncoder(sparse_output=False, categories='auto')
        label_onehot = ohe.fit_transform(y.reshape(-1, 1))

        combined = np.hstack([features, label_onehot])

        iso_forest = IsolationForest(
            n_estimators=200,
            contamination='auto',
            random_state=self.cfg.get('detection.random_state', 42),
            n_jobs=-1,
        )
        iso_forest.fit(combined)

        scores = iso_forest.decision_function(combined)
        scores = -scores
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return normalized.astype(np.float64)

    def ensemble_divergence(self, features: np.ndarray, labels: np.ndarray,
                            **kwargs) -> np.ndarray:
        """
        SVM+RF+MLP 三模型预测熵。

        训练三个不同类型的分类器，计算每个样本在三个模型上的预测概率熵。
        高熵表示模型间预测不一致，可能存在标签缺陷。

        Parameters
        ----------
        features : np.ndarray
            特征矩阵 (n_samples, n_features)
        labels : np.ndarray
            标签数组 (n_samples,)

        Returns
        -------
        np.ndarray
            缺陷置信度分数 (n_samples,)，值域 [0, 1]
        """
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        le = LabelEncoder()
        y = le.fit_transform(labels)
        n_classes = len(le.classes_)

        class_counts = np.bincount(y)
        min_count = int(np.min(class_counts))
        cv_folds = min(self.cfg.get('detection.cross_validation_folds', 5), min_count)
        if cv_folds < 2:
            return np.zeros(len(y), dtype=np.float64)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        random_state = self.cfg.get('detection.random_state', 42)

        svm = SVC(probability=True, random_state=random_state)
        rf = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=random_state,
        )

        models = [
            ("svm", svm),
            ("rf", rf),
            ("mlp", mlp),
        ]

        all_pred_probs = []
        for name, model in models:
            try:
                pred_probs = cross_val_predict(
                    model, X_scaled, y, cv=cv_folds, method='predict_proba'
                )
                all_pred_probs.append(pred_probs)
            except Exception:
                model.fit(X_scaled, y)
                all_pred_probs.append(model.predict_proba(X_scaled))

        avg_probs = np.mean(all_pred_probs, axis=0)
        avg_probs = np.clip(avg_probs, 1e-10, 1.0)
        avg_probs /= avg_probs.sum(axis=1, keepdims=True)

        entropy = -np.sum(avg_probs * np.log(avg_probs), axis=1)
        max_entropy = np.log(n_classes) if n_classes > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else entropy
        return normalized.astype(np.float64)

    def loss_trajectory(self, features: np.ndarray, labels: np.ndarray,
                        **kwargs) -> np.ndarray:
        """
        训练 ResNet-18（模拟），取交叉熵损失作为缺陷置信度。

        使用 sklearn 的 MLPClassifier 模拟深度网络训练过程，
        记录每个样本的交叉熵损失。高损失样本更可能存在标签缺陷。

        Parameters
        ----------
        features : np.ndarray
            特征矩阵 (n_samples, n_features)
        labels : np.ndarray
            标签数组 (n_samples,)

        Returns
        -------
        np.ndarray
            缺陷置信度分数 (n_samples,)，值域 [0, 1]
        """
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.neural_network import MLPClassifier

        le = LabelEncoder()
        y = le.fit_transform(labels)
        n_classes = len(le.classes_)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=200,
            random_state=self.cfg.get('detection.random_state', 42),
            warm_start=True,
        )
        mlp.fit(X_scaled, y)

        log_probs = mlp.predict_log_proba(X_scaled)
        sample_losses = -log_probs[np.arange(len(y)), y]

        normalized = (sample_losses - sample_losses.min()) / (sample_losses.max() - sample_losses.min() + 1e-8)
        return normalized.astype(np.float64)
