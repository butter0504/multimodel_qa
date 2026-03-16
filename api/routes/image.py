from fastapi import APIRouter, UploadFile, File
from modules.image_detector import ImageDetector

router = APIRouter()


@router.post("/detect")
async def detect_image(files: list[UploadFile] = File(...)):
    """检测图像数据质量"""
    # 读取上传的文件
    images = []
    for file in files:
        contents = await file.read()
        images.append(contents)
    
    # 检测数据质量
    detector = ImageDetector()
    result = detector.detect(images)
    
    return result
