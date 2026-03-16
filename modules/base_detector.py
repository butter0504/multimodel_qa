from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseDetector(ABC):
    """检测器基类，定义统一的检测接口"""

    @abstractmethod
    def detect(self, data: Any) -> Dict[str, Any]:
        """
        检测数据质量
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 检测结果
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        获取检测指标
        
        Returns:
            Dict[str, float]: 指标字典
        """
        pass

    @abstractmethod
    def get_issues(self) -> List[Dict[str, Any]]:
        """
        获取检测到的问题
        
        Returns:
            List[Dict[str, Any]]: 问题列表
        """
        pass
