import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from .base_detector import BaseDetector
from .cleanlab_wrapper import CleanlabWrapper
from typing import Dict, Any, List


class TextDetector(BaseDetector):
    """文本数据质量检测器"""
    
    def __init__(self):
        self.texts = None
        self.issues = []
        self.metrics = {}
    
    def detect(self, data: List[str]) -> Dict[str, Any]:
        """检测文本数据质量"""
        self.texts = data
        self.issues = []
        
        # 检测空文本
        empty_texts = [i for i, text in enumerate(data) if not text or text.strip() == ""]
        for idx in empty_texts:
            self.issues.append({
                "type": "empty_text",
                "index": idx
            })
        
        # 检测重复文本
        seen = {}
        duplicate_texts = []
        for i, text in enumerate(data):
            text_clean = text.strip()
            if text_clean in seen:
                duplicate_texts.append({
                    "type": "duplicate_text",
                    "index": i,
                    "original_index": seen[text_clean]
                })
            else:
                seen[text_clean] = i
        
        self.issues.extend(duplicate_texts)
        
        # 计算文本长度统计
        lengths = [len(text) for text in data]
        self.metrics = {
            "total_texts": len(data),
            "empty_text_rate": float(len(empty_texts) / len(data) * 100),
            "duplicate_text_rate": float(len(duplicate_texts) / len(data) * 100),
            "average_length": float(np.mean(lengths)) if lengths else 0,
            "min_length": float(np.min(lengths)) if lengths else 0,
            "max_length": float(np.max(lengths)) if lengths else 0
        }
        
        return {
            "metrics": self.metrics,
            "issues": self.issues
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """获取检测指标"""
        return self.metrics
    
    def get_issues(self) -> List[Dict[str, Any]]:
        """获取检测到的问题"""
        return self.issues
