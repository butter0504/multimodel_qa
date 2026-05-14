from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..data_object import DataObject


class ValidationResult:
    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None,
                 warnings: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class MetadataInfo:
    def __init__(self, num_classes: int = 0, sample_count: int = 0,
                 task_type: str = "classification",
                 label_names: Optional[List[str]] = None,
                 field_structure: Optional[Dict[str, str]] = None,
                 extra: Optional[Dict[str, Any]] = None):
        self.num_classes = num_classes
        self.sample_count = sample_count
        self.task_type = task_type
        self.label_names = label_names or []
        self.field_structure = field_structure or {}
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "sample_count": self.sample_count,
            "task_type": self.task_type,
            "label_names": self.label_names,
            "field_structure": self.field_structure,
            "extra": self.extra,
        }


class BaseAdapter(ABC):
    format_name: str = "base"
    supported_extensions: List[str] = []

    @abstractmethod
    def load(self, path: str, **kwargs) -> DataObject:
        pass

    @abstractmethod
    def validate(self, path: str, **kwargs) -> ValidationResult:
        pass

    @abstractmethod
    def get_metadata(self, path: str, **kwargs) -> MetadataInfo:
        pass

    def can_handle(self, path: str) -> bool:
        import os
        ext = os.path.splitext(path)[1].lower()
        return ext in self.supported_extensions

    def load_bytes(self, data: bytes, filename: str = "", **kwargs) -> DataObject:
        import tempfile
        import os
        suffix = os.path.splitext(filename)[1] if filename else ""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            tmp_path = f.name
        try:
            return self.load(tmp_path, **kwargs)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
