from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form

from modules.text_detector import TextDetector

router = APIRouter()


@router.post("/detect")
async def detect_text(
    file: UploadFile = File(...),
    labels: Optional[str] = Form(None),
    modules: Optional[str] = Form(None),
):
    """
    检测文本数据质量

    Parameters
    ----------
    file : UploadFile
        文本文件（TXT 或 CSV）
    labels : Optional[str]
        标签列表，以逗号分隔。如 "positive,negative,positive,..."
    modules : Optional[str]
        检测模块列表，以逗号分隔。可选值：
        text_length, duplicate, label_error, character_anomaly,
        language, perplexity, sentiment_consistency
        默认为 "text_length,duplicate,character_anomaly"
    """
    contents = await file.read()
    texts = contents.decode("utf-8").splitlines()
    texts = [t for t in texts if t.strip() != ""]

    label_list = None
    if labels:
        label_list = [l.strip() for l in labels.split(",") if l.strip()]
        if len(label_list) != len(texts):
            label_list = None

    module_list = None
    if modules:
        module_list = [m.strip() for m in modules.split(",") if m.strip()]

    detector = TextDetector()
    result = detector.detect(texts, labels=label_list, modules=module_list)

    return result
