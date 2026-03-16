import numpy as np
from cleanlab.classification import CleanLearning
from cleanlab.rank import get_label_quality_scores
from sklearn.base import BaseEstimator
from typing import Dict, List, Tuple, Any


class CleanlabWrapper:
    """Cleanlab 封装类，用于标签错误检测"""

    def __init__(self, model: BaseEstimator):
        """
        初始化 Cleanlab 封装器
        
        Args:
            model: 分类模型
        """
        self.model = model
        self.clean_learning = CleanLearning(self.model)
        self.issues = []

    def detect_label_issues(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        检测标签错误
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: 错误标签索引和检测结果
        """
        # 拟合模型并检测标签问题
        self.clean_learning.fit(X, y)
        
        # 获取标签错误的索引
        label_issues = self.clean_learning.get_label_issues()
        
        # 提取错误标签索引
        error_indices = label_issues[label_issues['is_label_issue']].index.tolist()
        
        # 计算标签质量分数
        # 先获取预测概率
        pred_probs = self.clean_learning.predict_proba(X)
        # 使用正确的函数获取标签质量分数
        label_quality_scores = get_label_quality_scores(y, pred_probs)
        
        # 构建结果
        result = {
            'error_indices': error_indices,
            'error_count': len(error_indices),
            'total_samples': len(y),
            'error_rate': len(error_indices) / len(y) if len(y) > 0 else 0,
            'label_quality_scores': label_quality_scores.tolist()
        }
        
        # 保存问题
        self.issues = [
            {
                'index': idx,
                'original_label': int(y[idx]),
                'predicted_label': int(self.clean_learning.predict(X[idx].reshape(1, -1))[0]),
                'label_quality_score': float(label_quality_scores[idx])
            }
            for idx in error_indices
        ]
        
        return error_indices, result

    def get_issues(self) -> List[Dict[str, Any]]:
        """
        获取检测到的标签问题
        
        Returns:
            List[Dict[str, Any]]: 问题列表
        """
        return self.issues

    def get_label_quality_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        获取标签质量分数
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            np.ndarray: 标签质量分数
        """
        # 获取预测概率
        pred_probs = self.clean_learning.predict_proba(X)
        # 使用正确的函数获取标签质量分数
        return get_label_quality_scores(y, pred_probs)
