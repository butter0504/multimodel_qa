from fastapi import APIRouter, UploadFile, File
import io
from modules.text_detector import TextDetector

router = APIRouter()


@router.post("/detect")
async def detect_text(file: UploadFile = File(...)):
    """检测文本数据质量"""
    # 读取上传的文件
    contents = await file.read()
    texts = contents.decode("utf-8").splitlines()
    
    # 检测数据质量
    detector = TextDetector()
    result = detector.detect(texts)
    
    return result
