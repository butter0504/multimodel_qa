import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .base_detector import BaseDetector
from .cleanlab_wrapper import CleanlabWrapper
from typing import Dict, Any, List


class TableDetector(BaseDetector):
    """表格数据质量检测器"""
    
    def __init__(self):
        self.data = None
        self.issues = []
        self.metrics = {}
    
    def detect(self, data: pd.DataFrame, label_col: str = None) -> Dict[str, Any]:
        """检测表格数据质量"""
        self.data = data
        self.issues = []
        
        # 基本信息
        basic_info = {
            "rows": len(data),
            "columns": len(data.columns),
            "columns_list": data.columns.tolist()
        }
        
        # 缺失值统计
        missing_values = data.isnull().sum()
        missing_total = missing_values.sum()
        missing_percentage = (missing_total / (len(data) * len(data.columns))) * 100
        missing_stats = {
            "total": int(missing_total),
            "percentage": float(missing_percentage),
            "per_column": {}
        }
        
        for col in data.columns:
            col_missing = int(missing_values[col])
            col_missing_percentage = (col_missing / len(data)) * 100 if len(data) > 0 else 0
            missing_stats["per_column"][col] = {
                "count": col_missing,
                "percentage": float(col_missing_percentage)
            }
            
            if col_missing > 0:
                self.issues.append({
                    "type": "missing_value",
                    "column": col,
                    "count": col_missing,
                    "percentage": float(col_missing_percentage)
                })
        
        # 异常值检测（IQR 方法）
        outliers = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = int(((data[col] < lower_bound) | (data[col] > upper_bound)).sum())
                if outlier_count > 0:
                    outliers[col] = {
                        "count": outlier_count,
                        "percentage": float(outlier_count / len(data) * 100),
                        "bounds": {
                            "lower": float(lower_bound),
                            "upper": float(upper_bound)
                        }
                    }
                    self.issues.append({
                        "type": "outlier",
                        "column": col,
                        "count": outlier_count,
                        "percentage": float(outlier_count / len(data) * 100)
                    })
        
        # 重复行检测
        duplicate_rows = data.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(data)) * 100 if len(data) > 0 else 0
        duplicate_stats = {
            "count": int(duplicate_rows),
            "percentage": float(duplicate_percentage)
        }
        
        if duplicate_rows > 0:
            self.issues.append({
                "type": "duplicate_rows",
                "count": int(duplicate_rows),
                "percentage": float(duplicate_percentage)
            })
        
        # 数据类型分析
        data_types = {}
        for col in data.columns:
            data_types[col] = str(data[col].dtype)
        
        # 标签错误检测（如果指定了 label_col）
        label_issues = {}
        if label_col and label_col in data.columns:
            # 使用完整的置信学习流程
            label_issues = self.detect_label_issues_with_confidence(data, label_col)
        
        # 计算指标
        self.metrics = {
            "rows": len(data),
            "columns": len(data.columns),
            "missing_value_rate": float(missing_percentage),
            "duplicate_row_rate": float(duplicate_percentage),
            "outlier_rate": float(sum(outliers.get(col, {}).get("count", 0) for col in outliers) / len(data) * 100) if len(data) > 0 else 0
        }
        
        return {
            "basic_info": basic_info,
            "missing_stats": missing_stats,
            "outlier_stats": outliers,
            "duplicate_stats": duplicate_stats,
            "data_types": data_types,
            "label_issues": label_issues,
            "metrics": self.metrics,
            "issues": self.issues
        }
    
    def detect_label_issues_with_confidence(self, df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
        """
        完整的置信学习标签错误检测流程
        
        步骤：
        1. 数据预处理：编码分类特征，处理缺失值
        2. 交叉验证获取无偏预测概率（关键！）
        3. 计算置信阈值（每个类别的百分位数）
        4. 构建置信联合矩阵 C_confident
        5. 估计噪声联合分布
        6. 计算质量分数并排序
        7. 返回问题样本
        """
        try:
            # 1. 数据预处理
            # 分离特征和标签
            X = df.drop(columns=[label_col]).copy()
            y = df[label_col].copy()
            
            # 处理分类特征
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                # 处理缺失值
                X[col] = X[col].fillna('missing')
                # 确保所有值都是字符串类型
                X[col] = X[col].astype(str)
                X[col] = le.fit_transform(X[col])
            
            # 处理数值特征的缺失值
            numeric_cols = X.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                X[col] = X[col].fillna(X[col].mean())
            
            # 标准化数值特征
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            
            # 确保 y 是数值型
            if not pd.api.types.is_numeric_dtype(y):
                y = pd.Categorical(y).codes
            
            y = y.values
            X = X.values
            
            # 2. 交叉验证获取无偏预测概率
            # 使用更复杂的模型以提高预测能力
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            # 使用交叉验证获取预测概率
            pred_probs = cross_val_predict(
                model, X, y, cv=5, method='predict_proba'
            )
            
            # 3. 计算置信阈值
            classes = np.unique(y)
            thresholds = {}
            for cls in classes:
                # 获取该类别的预测概率
                class_probs = pred_probs[y == cls, cls]
                if len(class_probs) > 0:
                    # 使用第85百分位作为阈值，降低阈值以检测更多潜在问题
                    thresholds[cls] = np.percentile(class_probs, 85)
                else:
                    thresholds[cls] = 0.5
            
            # 4. 构建置信联合矩阵 C_confident
            n_classes = len(classes)
            C_confident = np.zeros((n_classes, n_classes), dtype=int)
            
            for i, (prob, true_label) in enumerate(zip(pred_probs, y)):
                # 找到预测的类别
                pred_label = np.argmax(prob)
                # 检查是否超过阈值
                if prob[pred_label] >= thresholds.get(pred_label, 0.5):
                    C_confident[true_label, pred_label] += 1
            
            # 5. 估计噪声联合分布
            # 计算每个类别的先验概率
            class_counts = np.bincount(y, minlength=n_classes)
            prior = class_counts / len(y)
            
            # 估计噪声矩阵
            noise_matrix = C_confident / class_counts[:, None] if any(class_counts > 0) else np.zeros((n_classes, n_classes))
            
            # 6. 计算质量分数并排序
            # 使用 cleanlab 的方法计算标签质量分数
            from cleanlab.rank import get_label_quality_scores
            label_quality_scores = get_label_quality_scores(y, pred_probs)
            
            # 7. 识别问题样本
            # 降低阈值以检测更多潜在问题
            low_quality_threshold = 0.6
            error_indices = np.where(label_quality_scores < low_quality_threshold)[0].tolist()
            
            # 构建结果
            label_issues = {
                "error_count": len(error_indices),
                "error_rate": len(error_indices) / len(y) if len(y) > 0 else 0,
                "error_indices": error_indices,
                "label_quality_scores": label_quality_scores.tolist(),
                "confidence_thresholds": thresholds,
                "confident_joint_matrix": C_confident.tolist(),
                "noise_matrix": noise_matrix.tolist(),
                "debug_info": {
                    "classes": classes.tolist(),
                    "class_counts": class_counts.tolist(),
                    "prior": prior.tolist(),
                    "mean_quality_score": float(np.mean(label_quality_scores)),
                    "min_quality_score": float(np.min(label_quality_scores)),
                    "max_quality_score": float(np.max(label_quality_scores))
                }
            }
            
            # 添加到全局问题列表
            for idx in error_indices:
                self.issues.append({
                    "type": "label_error",
                    "index": int(idx),
                    "column": label_col,
                    "original_label": int(y[idx]),
                    "quality_score": float(label_quality_scores[idx])
                })
            
            return label_issues
            
        except Exception as e:
            return {
                "error": str(e)
            }
    
    def get_metrics(self) -> Dict[str, float]:
        """获取检测指标"""
        return self.metrics
    
    def get_issues(self) -> List[Dict[str, Any]]:
        """获取检测到的问题"""
        return self.issues
