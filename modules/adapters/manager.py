from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Type

from ..data_object import DataObject
from .base_adapter import BaseAdapter, MetadataInfo, ValidationResult
from .coco_adapter import COCOAdapter
from .csv_adapter import CSVAdapter
from .llm_adapter import LLMAdapter
from .voc_adapter import VOCAdapter
from .yolo_adapter import YOLOAdapter


class AdapterManager:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._adapters: Dict[str, BaseAdapter] = {}
        self._llm_adapter: Optional[LLMAdapter] = None
        self._register_default_adapters()

    def _register_default_adapters(self):
        for adapter_cls in [COCOAdapter, VOCAdapter, YOLOAdapter, CSVAdapter]:
            adapter = adapter_cls()
            self._adapters[adapter.format_name] = adapter
        self._llm_adapter = LLMAdapter(self.cfg)

    def register_adapter(self, adapter: BaseAdapter):
        self._adapters[adapter.format_name] = adapter

    def detect_format(self, path: str) -> Optional[str]:
        if not os.path.exists(path):
            return None

        if os.path.isdir(path):
            files = os.listdir(path)
            xml_count = sum(1 for f in files if f.lower().endswith(".xml"))
            txt_count = sum(1 for f in files if f.lower().endswith(".txt") and f != "classes.txt")
            img_count = sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))

            if xml_count > 0 and txt_count == 0:
                return "voc"
            if txt_count > 0 and xml_count == 0:
                classes_file = os.path.join(path, "classes.txt")
                if os.path.exists(classes_file) or img_count > 0:
                    return "yolo"

            subdirs = [d for d in files if os.path.isdir(os.path.join(path, d))]
            for subdir in ["Annotations", "JPEGImages", "ImageSets"]:
                if subdir in subdirs:
                    return "voc"
            for subdir in ["labels", "images"]:
                if subdir in subdirs:
                    return "yolo"

            return None

        ext = os.path.splitext(path)[1].lower()

        if ext == ".json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "annotations" in data and "images" in data:
                    return "coco"
                if isinstance(data, dict) and "categories" in data:
                    return "coco"
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            return None

        if ext == ".xml":
            return "voc"

        if ext == ".txt":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                parts = first_line.split()
                if len(parts) == 5:
                    try:
                        int(parts[0])
                        float(parts[1])
                        float(parts[2])
                        float(parts[3])
                        float(parts[4])
                        return "yolo"
                    except ValueError:
                        pass
            except Exception:
                pass
            return None

        if ext in [".csv", ".tsv"]:
            return "csv"

        return None

    def load(self, path: str, format_hint: Optional[str] = None,
             **kwargs) -> DataObject:
        fmt = format_hint or self.detect_format(path)

        if fmt and fmt in self._adapters:
            validation = self._adapters[fmt].validate(path, **kwargs)
            if not validation.is_valid:
                return DataObject(
                    source_format=fmt or "unknown",
                    metadata={
                        "load_status": "validation_failed",
                        "validation_errors": validation.errors,
                        "validation_warnings": validation.warnings,
                    },
                )
            return self._adapters[fmt].load(path, **kwargs)

        if self._llm_adapter is not None:
            llm_result = self._llm_adapter.load(path, **kwargs)
            if llm_result.metadata.get("parse_status") == "success":
                return llm_result

        return DataObject(
            source_format="unknown",
            metadata={
                "load_status": "no_adapter_found",
                "needs_human_review": True,
                "path": path,
            },
        )

    def load_bytes(self, data: bytes, filename: str = "",
                   format_hint: Optional[str] = None, **kwargs) -> DataObject:
        import tempfile

        fmt = format_hint
        if fmt is None and filename:
            ext = os.path.splitext(filename)[1].lower()
            fmt_map = {
                ".json": "coco",
                ".xml": "voc",
                ".txt": "yolo",
                ".csv": "csv",
                ".tsv": "csv",
            }
            fmt = fmt_map.get(ext)

        if fmt and fmt in self._adapters:
            return self._adapters[fmt].load_bytes(data, filename, **kwargs)

        if self._llm_adapter is not None:
            raw_data = data.decode("utf-8", errors="replace")
            llm_result = self._llm_adapter.load(
                "", raw_data=raw_data, **kwargs
            )
            if llm_result.metadata.get("parse_status") == "success":
                return llm_result

        return DataObject(
            source_format="unknown",
            metadata={
                "load_status": "no_adapter_found",
                "needs_human_review": True,
                "filename": filename,
            },
        )

    def validate(self, path: str, format_hint: Optional[str] = None,
                 **kwargs) -> ValidationResult:
        fmt = format_hint or self.detect_format(path)

        if fmt and fmt in self._adapters:
            return self._adapters[fmt].validate(path, **kwargs)

        return ValidationResult(False, errors=[f"无法识别数据格式: {path}"])

    def get_metadata(self, path: str, format_hint: Optional[str] = None,
                     **kwargs) -> MetadataInfo:
        fmt = format_hint or self.detect_format(path)

        if fmt and fmt in self._adapters:
            return self._adapters[fmt].get_metadata(path, **kwargs)

        return MetadataInfo()

    def get_supported_formats(self) -> List[str]:
        return list(self._adapters.keys()) + ["llm"]

    def get_adapter(self, format_name: str) -> Optional[BaseAdapter]:
        if format_name == "llm":
            return self._llm_adapter
        return self._adapters.get(format_name)
