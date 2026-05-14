from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form

from modules.interface import Interface, DetectionRequest

router = APIRouter()
_interface = Interface()


@router.post("/detect")
async def detect_text(
    file: UploadFile = File(...),
    labels: Optional[str] = Form(None),
    modules: Optional[str] = Form(None),
    format_hint: Optional[str] = Form(None),
    label_column: Optional[str] = Form(None),
    text_column: Optional[str] = Form(None),
):
    contents = await file.read()

    label_list = None
    if labels:
        label_list = [l.strip() for l in labels.split(",") if l.strip()]

    module_list = None
    if modules:
        module_list = [m.strip() for m in modules.split(",") if m.strip()]

    request = DetectionRequest(
        source=contents,
        source_type="bytes",
        modality="text",
        modules=module_list or ["text_length", "duplicate", "character_anomaly"],
        format_hint=format_hint,
        label_column=label_column,
        text_column=text_column,
        extra={"filename": file.filename, "labels": label_list},
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


@router.post("/detect/file")
async def detect_text_file(
    file: UploadFile = File(...),
    modules: Optional[str] = Form(None),
    format_hint: Optional[str] = Form(None),
    label_column: Optional[str] = Form(None),
    text_column: Optional[str] = Form(None),
):
    import tempfile
    import os

    contents = await file.read()
    suffix = os.path.splitext(file.filename)[1] if file.filename else ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(contents)
        tmp_path = f.name

    try:
        module_list = None
        if modules:
            module_list = [m.strip() for m in modules.split(",") if m.strip()]

        request = DetectionRequest(
            source=tmp_path,
            source_type="path",
            modality="text",
            modules=module_list or ["text_length", "duplicate", "character_anomaly"],
            format_hint=format_hint,
            label_column=label_column,
            text_column=text_column,
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
