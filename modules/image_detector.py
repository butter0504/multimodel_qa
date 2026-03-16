import cv2
import numpy as np
from PIL import Image
import io
from .base_detector import BaseDetector
from typing import Dict, Any, List


class ImageDetector(BaseDetector):
    """图像数据质量检测器"""
    
    def __init__(self):
        self.images = None
        self.issues = []
        self.metrics = {}
    
    def detect(self, data: List[bytes]) -> Dict[str, Any]:
        """检测图像数据质量"""
        self.images = data
        self.issues = []
        
        # 检测图像质量
        for i, img_bytes in enumerate(data):
            try:
                # 读取图像
                img = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img)
                
                # 检测图像大小
                height, width = img_array.shape[:2]
                if height < 100 or width < 100:
                    self.issues.append({
                        "type": "small_image",
                        "index": i,
                        "height": height,
                        "width": width
                    })
                
                # 检测图像清晰度（使用 Laplacian 方差）
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < 50:
                    self.issues.append({
                        "type": "blurry_image",
                        "index": i,
                        "sharpness": float(laplacian_var)
                    })
                
            except Exception as e:
                self.issues.append({
                    "type": "corrupted_image",
                    "index": i,
                    "error": str(e)
                })
        
        # 计算指标
        self.metrics = {
            "total_images": len(data),
            "issue_rate": float(len(self.issues) / len(data) * 100) if data else 0
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
