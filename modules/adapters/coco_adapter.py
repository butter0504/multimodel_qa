from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

from ..data_object import Annotation, BBox, DataObject, Sample
from .base_adapter import BaseAdapter, MetadataInfo, ValidationResult


class COCOAdapter(BaseAdapter):
    format_name = "coco"
    supported_extensions = [".json"]

    def load(self, path: str, **kwargs) -> DataObject:
        image_dir = kwargs.get("image_dir", None)
        load_images = kwargs.get("load_images", True)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if image_dir is None:
            image_dir = os.path.dirname(path)

        categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
        label_names = list(categories.values())

        img_info = {img["id"]: img for img in data.get("images", [])}

        ann_by_image: Dict[int, List[Annotation]] = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            cat_id = ann.get("category_id")
            label = categories.get(cat_id, str(cat_id)) if cat_id is not None else "unknown"

            bbox = None
            if "bbox" in ann and ann["bbox"] is not None:
                x, y, w, h = ann["bbox"]
                bbox = BBox.from_xywh(x, y, w, h)

            annotation = Annotation(
                label=label,
                bbox=bbox,
                confidence=ann.get("score", 1.0),
                extra={"coco_id": ann.get("id"), "area": ann.get("area"), "iscrowd": ann.get("iscrowd", 0)},
            )
            ann_by_image.setdefault(img_id, []).append(annotation)

        samples = []
        for img_id, img_data in img_info.items():
            file_name = img_data.get("file_name", "")
            sample_id = str(img_id)

            sample = Sample(
                sample_id=sample_id,
                file_path=file_name,
                annotations=ann_by_image.get(img_id, []),
                extra={
                    "width": img_data.get("width"),
                    "height": img_data.get("height"),
                    "coco_url": img_data.get("coco_url", ""),
                },
            )

            if load_images and image_dir:
                full_path = os.path.join(image_dir, file_name)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, "rb") as img_f:
                            sample.data = img_f.read()
                    except Exception:
                        pass

            samples.append(sample)

        has_bbox = any(ann.bbox is not None for s in samples for ann in s.annotations)
        task_type = "detection" if has_bbox else "classification"

        return DataObject(
            samples=samples,
            label_names=label_names,
            task_type=task_type,
            source_format="coco",
            metadata={
                "source_file": path,
                "num_images": len(img_info),
                "num_annotations": len(data.get("annotations", [])),
                "num_categories": len(categories),
            },
        )

    def validate(self, path: str, **kwargs) -> ValidationResult:
        errors = []
        warnings = []

        if not os.path.exists(path):
            return ValidationResult(False, errors=[f"文件不存在: {path}"])

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return ValidationResult(False, errors=[f"JSON 解析失败: {str(e)}"])

        if not isinstance(data, dict):
            errors.append("COCO 格式要求顶层为 JSON 对象")
            return ValidationResult(False, errors=errors)

        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            if key not in data:
                errors.append(f"缺少必需字段: {key}")

        if errors:
            return ValidationResult(False, errors=errors)

        if not isinstance(data["images"], list):
            errors.append("images 字段应为数组")
        else:
            for i, img in enumerate(data["images"]):
                if "id" not in img:
                    errors.append(f"images[{i}] 缺少 id 字段")
                if "file_name" not in img:
                    warnings.append(f"images[{i}] 缺少 file_name 字段")

        if not isinstance(data["categories"], list):
            errors.append("categories 字段应为数组")
        else:
            for i, cat in enumerate(data["categories"]):
                if "id" not in cat:
                    errors.append(f"categories[{i}] 缺少 id 字段")
                if "name" not in cat:
                    errors.append(f"categories[{i}] 缺少 name 字段")

        if not isinstance(data["annotations"], list):
            errors.append("annotations 字段应为数组")
        else:
            for i, ann in enumerate(data["annotations"]):
                if "image_id" not in ann:
                    errors.append(f"annotations[{i}] 缺少 image_id 字段")
                if "category_id" not in ann:
                    errors.append(f"annotations[{i}] 缺少 category_id 字段")
                if "bbox" in ann and ann["bbox"] is not None:
                    if not isinstance(ann["bbox"], list) or len(ann["bbox"]) != 4:
                        errors.append(f"annotations[{i}] bbox 格式错误，应为 [x, y, w, h]")
                    else:
                        x, y, w, h = ann["bbox"]
                        if w <= 0 or h <= 0:
                            warnings.append(f"annotations[{i}] bbox 宽度或高度非正: [{x}, {y}, {w}, {h}]")

        img_ids = {img["id"] for img in data.get("images", []) if "id" in img}
        cat_ids = {cat["id"] for cat in data.get("categories", []) if "id" in cat}

        for i, ann in enumerate(data.get("annotations", [])):
            if "image_id" in ann and ann["image_id"] not in img_ids:
                warnings.append(f"annotations[{i}] 的 image_id={ann['image_id']} 在 images 中不存在")
            if "category_id" in ann and ann["category_id"] not in cat_ids:
                warnings.append(f"annotations[{i}] 的 category_id={ann['category_id']} 在 categories 中不存在")

        return ValidationResult(len(errors) == 0, errors=errors, warnings=warnings)

    def get_metadata(self, path: str, **kwargs) -> MetadataInfo:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        label_names = [cat["name"] for cat in categories if "name" in cat]
        images = data.get("images", [])
        annotations = data.get("annotations", [])

        has_bbox = any("bbox" in ann and ann["bbox"] is not None for ann in annotations)
        task_type = "detection" if has_bbox else "classification"

        field_structure = {}
        if images:
            field_structure["images"] = "id, file_name, width, height"
        if annotations:
            field_structure["annotations"] = "id, image_id, category_id, bbox"
        if categories:
            field_structure["categories"] = "id, name, supercategory"

        return MetadataInfo(
            num_classes=len(label_names),
            sample_count=len(images),
            task_type=task_type,
            label_names=label_names,
            field_structure=field_structure,
            extra={
                "num_annotations": len(annotations),
                "has_bbox": has_bbox,
            },
        )
