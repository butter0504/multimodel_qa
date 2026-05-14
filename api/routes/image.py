from fastapi import APIRouter, UploadFile, File, Form, Query
from typing import Optional, List

from modules.interface import Interface, DetectionRequest

router = APIRouter()
_interface = Interface()


@router.post("/detect")
async def detect_image(
    files: List[UploadFile] = File(...),
    modules: Optional[str] = Form(None),
    format_hint: Optional[str] = Form(None),
):
    images = []
    for file in files:
        contents = await file.read()
        images.append(contents)

    module_list = None
    if modules:
        module_list = [m.strip() for m in modules.split(",") if m.strip()]

    if len(images) == 1:
        request = DetectionRequest(
            source=images[0],
            source_type="bytes",
            modality="image",
            modules=module_list or ["basic_quality"],
            format_hint=format_hint,
            extra={"filename": files[0].filename},
        )
    else:
        request = DetectionRequest(
            source=images,
            source_type="bytes_list",
            modality="image",
            modules=module_list or ["basic_quality"],
            format_hint=format_hint,
            extra={"filename": files[0].filename if files else ""},
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
async def detect_image_file(
    file: UploadFile = File(...),
    modules: Optional[str] = Form(None),
    format_hint: Optional[str] = Form(None),
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
            modality="image",
            modules=module_list or ["basic_quality"],
            format_hint=format_hint,
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


@router.post("/validate")
async def validate_data(
    file: UploadFile = File(...),
    format_hint: Optional[str] = Form(None),
):
    import tempfile
    import os

    contents = await file.read()
    suffix = os.path.splitext(file.filename)[1] if file.filename else ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(contents)
        tmp_path = f.name

    try:
        result = _interface.validate_data(tmp_path, format_hint)
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/formats")
def get_supported_formats():
    return {"formats": _interface.get_supported_formats()}
