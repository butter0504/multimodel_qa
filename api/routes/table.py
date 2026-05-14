from fastapi import APIRouter, UploadFile, File, Query
from typing import Optional

from modules.interface import Interface, DetectionRequest

router = APIRouter()
_interface = Interface()


@router.post("/detect")
async def detect_table(
    file: UploadFile = File(...),
    label_col: Optional[str] = Query(None, description="标签列名称"),
    format_hint: Optional[str] = Query(None, description="数据格式提示"),
):
    import tempfile
    import os

    contents = await file.read()
    suffix = os.path.splitext(file.filename)[1] if file.filename else ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(contents)
        tmp_path = f.name

    try:
        request = DetectionRequest(
            source=tmp_path,
            source_type="path",
            modality="table",
            modules=["missing_values", "outliers", "duplicates"],
            format_hint=format_hint or "csv",
            label_column=label_col,
            extra={"filename": file.filename},
        )

        response = _interface.process_request(request)

        result = {
            "success": response.success,
            "data_info": response.data_object_info,
            "detection": response.detection_result,
        }
        if response.errors:
            result["errors"] = response.errors
        if response.warnings:
            result["warnings"] = response.warnings

        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
