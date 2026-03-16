from fastapi import APIRouter, UploadFile, File, Query
import pandas as pd
import io
import numpy as np
from modules.table_detector import TableDetector
from api.models.schemas import TableAnalysisResponse

router = APIRouter()


def convert_numpy(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    else:
        return obj


@router.post("/detect", response_model=TableAnalysisResponse)
async def detect_table(
    file: UploadFile = File(...),
    label_col: str = Query(None, description="标签列名称")
):
    """检测表格数据质量"""
    # 读取上传的文件
    contents = await file.read()
    
    # 根据文件扩展名读取数据
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        return {"error": "Unsupported file format. Please upload a CSV or Excel file."}
    
    # 检测数据质量
    detector = TableDetector()
    result = detector.detect(df, label_col=label_col)
    
    # 添加文件名到结果
    result["basic_info"]["filename"] = file.filename
    
    # 转换 numpy 类型为 Python 原生类型
    result = convert_numpy(result)
    
    return result
