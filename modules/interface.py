from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .adapters.manager import AdapterManager
from .config import Config
from .data_object import DataObject
from .scheduler import DetectionTask, Scheduler


@dataclass
class DetectionRequest:
    source: Any = None
    source_type: str = "auto"
    format_hint: Optional[str] = None
    modality: Optional[str] = None
    modules: Optional[List[str]] = None
    label_column: Optional[str] = None
    text_column: Optional[str] = None
    filename_column: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResponse:
    success: bool = True
    data_object_info: Dict[str, Any] = field(default_factory=dict)
    detection_result: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Interface:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self._adapter_manager = AdapterManager(self.cfg)
        self._scheduler = Scheduler(self.cfg)

    def process_request(self, request: DetectionRequest) -> DetectionResponse:
        response = DetectionResponse()

        validation_errors = self._validate_request(request)
        if validation_errors:
            response.success = False
            response.errors = validation_errors
            return response

        data_obj = self._load_data(request)
        if data_obj is None:
            response.success = False
            response.errors = ["数据加载失败"]
            return response

        if data_obj.metadata.get("load_status") == "validation_failed":
            response.success = False
            response.errors = data_obj.metadata.get("validation_errors", ["数据校验失败"])
            response.warnings = data_obj.metadata.get("validation_warnings", [])
            return response

        if data_obj.metadata.get("load_status") == "no_adapter_found":
            response.success = False
            response.errors = ["无法识别数据格式，请指定 format_hint 或使用支持的格式"]
            return response

        response.data_object_info = data_obj.to_dict()

        if data_obj.metadata.get("needs_human_review"):
            response.warnings.append("部分数据需要人工审核，已标记")

        task = self._scheduler.create_task(
            data_obj=data_obj,
            modules=request.modules,
            extra={
                "label_column": request.label_column,
                "text_column": request.text_column,
                "filename_column": request.filename_column,
                **request.extra,
            },
        )

        if request.modality and request.modality != task.modality:
            task.modality = request.modality

        completed_task = self._scheduler.submit(task)

        if completed_task.status == "failed":
            response.success = False
            response.errors = [completed_task.error or "检测任务执行失败"]
            return response

        response.detection_result = completed_task.result or {}

        return response

    def process_batch(self, requests: List[DetectionRequest],
                      max_workers: int = 2) -> List[DetectionResponse]:
        tasks = []
        data_objects = []

        for req in requests:
            validation_errors = self._validate_request(req)
            if validation_errors:
                continue

            data_obj = self._load_data(req)
            if data_obj is None:
                continue

            task = self._scheduler.create_task(
                data_obj=data_obj,
                modules=req.modules,
                extra={
                    "label_column": req.label_column,
                    "text_column": req.text_column,
                    **req.extra,
                },
            )
            tasks.append(task)
            data_objects.append((req, data_obj))

        completed_tasks = self._scheduler.submit_batch(tasks, max_workers=max_workers)

        results = []
        for (req, data_obj), task in zip(data_objects, completed_tasks):
            response = DetectionResponse()
            response.data_object_info = data_obj.to_dict()

            if task.status == "failed":
                response.success = False
                response.errors = [task.error or "检测任务执行失败"]
            else:
                response.detection_result = task.result or {}

            results.append(response)

        return results

    def _validate_request(self, request: DetectionRequest) -> List[str]:
        errors = []

        if request.source is None:
            errors.append("缺少数据源 (source)")

        if request.modules is not None:
            valid_modules = {
                "image": ["basic_quality", "label_error", "distribution_shift", "uncertainty"],
                "text": ["text_length", "duplicate", "label_error", "character_anomaly",
                         "language", "perplexity", "sentiment_consistency"],
                "table": ["missing_values", "outliers", "duplicates", "label_error"],
            }
            if request.modality and request.modality in valid_modules:
                invalid = [m for m in request.modules if m not in valid_modules[request.modality]]
                if invalid:
                    errors.append(f"模块 {invalid} 不适用于 {request.modality} 模态")

        return errors

    def _load_data(self, request: DetectionRequest) -> Optional[DataObject]:
        source = request.source

        if isinstance(source, str):
            return self._adapter_manager.load(
                path=source,
                format_hint=request.format_hint,
                label_column=request.label_column,
                text_column=request.text_column,
                filename_column=request.filename_column,
            )

        if isinstance(source, bytes):
            filename = request.extra.get("filename", "")
            return self._adapter_manager.load_bytes(
                data=source,
                filename=filename,
                format_hint=request.format_hint,
                label_column=request.label_column,
                text_column=request.text_column,
            )

        if isinstance(source, dict):
            return self._load_from_dict(source, request)

        if isinstance(source, DataObject):
            return source

        if isinstance(source, list):
            return self._load_from_list(source, request)

        return None

    def _load_from_dict(self, data: Dict[str, Any],
                        request: DetectionRequest) -> DataObject:
        from .data_object import Annotation, Sample

        samples = []
        for i, item in enumerate(data.get("items", data.get("samples", []))):
            if isinstance(item, dict):
                sample_id = str(item.get("id", item.get("sample_id", i)))
                label = item.get("label", item.get("category", None))
                annotations = []
                if label is not None:
                    annotations.append(Annotation(label=str(label)))
                samples.append(Sample(
                    sample_id=sample_id,
                    annotations=annotations,
                    extra={k: v for k, v in item.items() if k not in ("id", "sample_id", "label", "category")},
                ))
            else:
                samples.append(Sample(sample_id=str(i), annotations=[Annotation(label=str(item))]))

        label_names = data.get("label_names", [])
        if not label_names:
            label_names = sorted(set(
                ann.label for s in samples for ann in s.annotations if ann.label
            ))

        return DataObject(
            samples=samples,
            label_names=label_names,
            task_type=data.get("task_type", "classification"),
            source_format="dict",
            metadata=data.get("metadata", {}),
        )

    def _load_from_list(self, data: list, request: DetectionRequest) -> DataObject:
        from .data_object import Annotation, Sample

        samples = []
        for i, item in enumerate(data):
            if isinstance(item, bytes):
                samples.append(Sample(sample_id=str(i), data=item))
            elif isinstance(item, str):
                samples.append(Sample(
                    sample_id=str(i),
                    data=item.encode("utf-8"),
                    annotations=[],
                ))
            elif isinstance(item, dict):
                label = item.get("label", None)
                annotations = []
                if label is not None:
                    annotations.append(Annotation(label=str(label)))
                samples.append(Sample(
                    sample_id=str(i),
                    annotations=annotations,
                    extra=item,
                ))
            else:
                samples.append(Sample(
                    sample_id=str(i),
                    annotations=[Annotation(label=str(item))],
                ))

        return DataObject(
            samples=samples,
            source_format="list",
            metadata={"modality": request.modality or "unknown"},
        )

    def get_supported_formats(self) -> List[str]:
        return self._adapter_manager.get_supported_formats()

    def validate_data(self, path: str, format_hint: Optional[str] = None) -> Dict[str, Any]:
        result = self._adapter_manager.validate(path, format_hint)
        return result.to_dict()

    def get_data_metadata(self, path: str, format_hint: Optional[str] = None) -> Dict[str, Any]:
        metadata = self._adapter_manager.get_metadata(path, format_hint)
        return metadata.to_dict()
