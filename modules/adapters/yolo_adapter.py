from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..data_object import Annotation, BBox, DataObject, Sample
from .base_adapter import BaseAdapter, MetadataInfo, ValidationResult


class YOLOAdapter(BaseAdapter):
    format_name = "yolo"
    supported_extensions = [".txt"]

    def load(self, path: str, **kwargs) -> DataObject:
        image_dir = kwargs.get("image_dir", None)
        class_file = kwargs.get("class_file", None)
        load_images = kwargs.get("load_images", True)

        if os.path.isdir(path):
            return self._load_from_directory(path, image_dir, class_file, load_images)

        return self._load_single_label(path, image_dir, class_file, load_images)

    def _load_from_directory(self, label_dir: str, image_dir: Optional[str],
                             class_file: Optional[str], load_images: bool) -> DataObject:
        if image_dir is None:
            for candidate in ["images", "JPEGImages"]:
                candidate_dir = os.path.join(os.path.dirname(label_dir), candidate)
                if os.path.exists(candidate_dir):
                    image_dir = candidate_dir
                    break

        label_names = self._load_class_names(label_dir, class_file)

        txt_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(".txt") and f != "classes.txt"])

        samples = []
        for txt_file in txt_files:
            txt_path = os.path.join(label_dir, txt_file)
            sample = self._parse_yolo_label(txt_path, image_dir, label_names, load_images)
            if sample is not None:
                samples.append(sample)

        has_bbox = any(ann.bbox is not None for s in samples for ann in s.annotations)
        task_type = "detection" if has_bbox else "classification"

        return DataObject(
            samples=samples,
            label_names=label_names,
            task_type=task_type,
            source_format="yolo",
            metadata={
                "label_dir": label_dir,
                "image_dir": image_dir,
                "num_samples": len(samples),
            },
        )

    def _load_single_label(self, txt_path: str, image_dir: Optional[str],
                           class_file: Optional[str], load_images: bool) -> DataObject:
        label_dir = os.path.dirname(txt_path)
        label_names = self._load_class_names(label_dir, class_file)

        sample = self._parse_yolo_label(txt_path, image_dir, label_names, load_images)
        if sample is None:
            return DataObject(source_format="yolo")

        has_bbox = any(ann.bbox is not None for ann in sample.annotations)
        task_type = "detection" if has_bbox else "classification"

        return DataObject(
            samples=[sample],
            label_names=label_names,
            task_type=task_type,
            source_format="yolo",
            metadata={"source_file": txt_path},
        )

    def _parse_yolo_label(self, txt_path: str, image_dir: Optional[str],
                          label_names: List[str], load_images: bool) -> Optional[Sample]:
        sample_id = os.path.splitext(os.path.basename(txt_path))[0]

        annotations = []
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    except (ValueError, IndexError):
                        continue

                    label = label_names[class_id] if class_id < len(label_names) else str(class_id)

                    bbox = BBox.from_yolo(cx, cy, w, h)
                    annotations.append(Annotation(
                        label=label,
                        bbox=bbox,
                        extra={"class_id": class_id, "yolo_format": [cx, cy, w, h]},
                    ))
        except Exception:
            return None

        image_data = None
        file_path = None
        if image_dir:
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                candidate = os.path.join(image_dir, sample_id + ext)
                if os.path.exists(candidate):
                    file_path = candidate
                    if load_images:
                        try:
                            with open(candidate, "rb") as f:
                                image_data = f.read()
                        except Exception:
                            pass
                    break

        return Sample(
            sample_id=sample_id,
            data=image_data,
            file_path=file_path,
            annotations=annotations,
        )

    def _load_class_names(self, label_dir: str, class_file: Optional[str] = None) -> List[str]:
        if class_file and os.path.exists(class_file):
            with open(class_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        classes_path = os.path.join(label_dir, "classes.txt")
        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        parent_dir = os.path.dirname(label_dir)
        for candidate in ["classes.txt", "labels.txt", "names.txt"]:
            candidate_path = os.path.join(parent_dir, candidate)
            if os.path.exists(candidate_path):
                with open(candidate_path, "r", encoding="utf-8") as f:
                    return [line.strip() for line in f if line.strip()]

        return []

    def validate(self, path: str, **kwargs) -> ValidationResult:
        errors = []
        warnings = []

        if os.path.isdir(path):
            txt_files = [f for f in os.listdir(path) if f.lower().endswith(".txt") and f != "classes.txt"]
            if not txt_files:
                errors.append("目录中没有找到 YOLO 标签文件（.txt）")
                return ValidationResult(False, errors=errors)

            class_file = kwargs.get("class_file", None)
            label_names = self._load_class_names(path, class_file)
            if not label_names:
                warnings.append("未找到 classes.txt 类别名称文件，将使用数字 ID 作为类别名")

            for fname in txt_files[:100]:
                txt_path = os.path.join(path, fname)
                result = self._validate_single_label(txt_path, label_names)
                errors.extend(result.errors)
                warnings.extend(result.warnings)

            return ValidationResult(len(errors) == 0, errors=errors, warnings=warnings)

        label_dir = os.path.dirname(path)
        label_names = self._load_class_names(label_dir, kwargs.get("class_file"))
        return self._validate_single_label(path, label_names)

    def _validate_single_label(self, txt_path: str, label_names: List[str]) -> ValidationResult:
        errors = []
        warnings = []

        if not os.path.exists(txt_path):
            return ValidationResult(False, errors=[f"文件不存在: {txt_path}"])

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            return ValidationResult(False, errors=[f"文件读取失败: {str(e)}"])

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                errors.append(f"行 {line_num}: 字段不足，YOLO 格式需要至少 5 个字段 (class cx cy w h)")
                continue

            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                errors.append(f"行 {line_num}: 数值解析失败")
                continue

            if cx < 0 or cx > 1:
                warnings.append(f"行 {line_num}: 中心点 x={cx} 超出 [0,1] 范围")
            if cy < 0 or cy > 1:
                warnings.append(f"行 {line_num}: 中心点 y={cy} 超出 [0,1] 范围")
            if w <= 0 or w > 1:
                errors.append(f"行 {line_num}: 宽度 w={w} 不在 (0,1] 范围")
            if h <= 0 or h > 1:
                errors.append(f"行 {line_num}: 高度 h={h} 不在 (0,1] 范围")

            if label_names and class_id >= len(label_names):
                warnings.append(f"行 {line_num}: 类别 ID {class_id} 超出类别名称列表范围")

        return ValidationResult(len(errors) == 0, errors=errors, warnings=warnings)

    def get_metadata(self, path: str, **kwargs) -> MetadataInfo:
        if os.path.isdir(path):
            class_file = kwargs.get("class_file", None)
            label_names = self._load_class_names(path, class_file)

            txt_files = [f for f in os.listdir(path) if f.lower().endswith(".txt") and f != "classes.txt"]

            return MetadataInfo(
                num_classes=len(label_names),
                sample_count=len(txt_files),
                task_type="detection",
                label_names=label_names,
                field_structure={"label": "class_id center_x center_y width height"},
                extra={"has_classes_file": len(label_names) > 0},
            )

        label_dir = os.path.dirname(path)
        label_names = self._load_class_names(label_dir, kwargs.get("class_file"))

        return MetadataInfo(
            num_classes=len(label_names),
            sample_count=1,
            task_type="detection",
            label_names=label_names,
            field_structure={"label": "class_id center_x center_y width height"},
        )
