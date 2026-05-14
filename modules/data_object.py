from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_list(self) -> List[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def width(self) -> float:
        return self.x_max - self.x_min

    def height(self) -> float:
        return self.y_max - self.y_min

    def area(self) -> float:
        return max(0.0, self.width()) * max(0.0, self.height())

    def is_valid(self) -> bool:
        return self.x_max > self.x_min and self.y_max > self.y_min

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> 'BBox':
        return cls(x_min=x, y_min=y, x_max=x + w, y_max=y + h)

    @classmethod
    def from_yolo(cls, cx: float, cy: float, w: float, h: float,
                  img_w: float = 1.0, img_h: float = 1.0) -> 'BBox':
        abs_cx = cx * img_w
        abs_cy = cy * img_h
        abs_w = w * img_w
        abs_h = h * img_h
        return cls(
            x_min=abs_cx - abs_w / 2,
            y_min=abs_cy - abs_h / 2,
            x_max=abs_cx + abs_w / 2,
            y_max=abs_cy + abs_h / 2,
        )


@dataclass
class Annotation:
    label: str
    bbox: Optional[BBox] = None
    confidence: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"label": self.label, "confidence": self.confidence}
        if self.bbox is not None:
            d["bbox"] = self.bbox.to_list()
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass
class Sample:
    sample_id: str
    data: Optional[bytes] = None
    file_path: Optional[str] = None
    annotations: List[Annotation] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> List[str]:
        return [ann.label for ann in self.annotations]

    @property
    def primary_label(self) -> Optional[str]:
        if self.annotations:
            best = max(self.annotations, key=lambda a: a.confidence)
            return best.label
        return None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"sample_id": self.sample_id}
        if self.file_path is not None:
            d["file_path"] = self.file_path
        d["annotations"] = [ann.to_dict() for ann in self.annotations]
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass
class DataObject:
    samples: List[Sample] = field(default_factory=list)
    label_names: List[str] = field(default_factory=list)
    task_type: str = "classification"
    source_format: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    flagged_samples: List[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    @property
    def has_labels(self) -> bool:
        return any(s.annotations for s in self.samples)

    @property
    def has_bboxes(self) -> bool:
        return any(
            ann.bbox is not None
            for s in self.samples
            for ann in s.annotations
        )

    def get_images(self) -> List[bytes]:
        images = []
        for s in self.samples:
            if s.data is not None:
                images.append(s.data)
        return images

    def get_labels(self) -> List[Any]:
        labels = []
        for s in self.samples:
            if s.primary_label is not None:
                labels.append(s.primary_label)
            else:
                labels.append(None)
        return labels

    def get_label_indices(self) -> List[int]:
        label_to_idx = {name: idx for idx, name in enumerate(self.label_names)}
        indices = []
        for s in self.samples:
            pl = s.primary_label
            if pl is not None and pl in label_to_idx:
                indices.append(label_to_idx[pl])
            else:
                indices.append(-1)
        return indices

    def get_filenames(self) -> List[str]:
        return [s.file_path or s.sample_id for s in self.samples]

    def flag_sample(self, sample_id: str, reason: str = ""):
        if sample_id not in self.flagged_samples:
            self.flagged_samples.append(sample_id)
        for s in self.samples:
            if s.sample_id == sample_id:
                s.extra["flagged"] = True
                s.extra["flag_reason"] = reason
                break

    def get_valid_samples(self) -> List[Sample]:
        flagged = set(self.flagged_samples)
        return [s for s in self.samples if s.sample_id not in flagged]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "num_classes": self.num_classes,
            "task_type": self.task_type,
            "source_format": self.source_format,
            "label_names": self.label_names,
            "has_labels": self.has_labels,
            "has_bboxes": self.has_bboxes,
            "flagged_count": len(self.flagged_samples),
            "metadata": self.metadata,
            "samples": [s.to_dict() for s in self.samples[:10]],
        }

    def to_legacy_format(self) -> Dict[str, Any]:
        images = self.get_images()
        labels = self.get_labels()
        filenames = self.get_filenames()
        return {
            "images": images,
            "labels": labels if any(l is not None for l in labels) else None,
            "label_names": self.label_names,
            "filenames": filenames,
            "metadata": {
                **self.metadata,
                "source": self.source_format,
                "total_samples": self.sample_count,
                "num_classes": self.num_classes,
                "task_type": self.task_type,
            },
            "source_type": self.source_format,
            "has_labels": self.has_labels,
        }
