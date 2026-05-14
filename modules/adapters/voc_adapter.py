from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from ..data_object import Annotation, BBox, DataObject, Sample
from .base_adapter import BaseAdapter, MetadataInfo, ValidationResult


class VOCAdapter(BaseAdapter):
    format_name = "voc"
    supported_extensions = [".xml"]

    def load(self, path: str, **kwargs) -> DataObject:
        image_dir = kwargs.get("image_dir", None)
        load_images = kwargs.get("load_images", True)
        annotation_dir = kwargs.get("annotation_dir", None)

        if os.path.isdir(path):
            return self._load_from_directory(path, image_dir, load_images)

        return self._load_single_annotation(path, image_dir, load_images)

    def _load_from_directory(self, ann_dir: str, image_dir: Optional[str],
                             load_images: bool) -> DataObject:
        if image_dir is None:
            image_dir = os.path.join(os.path.dirname(ann_dir), "JPEGImages")
            if not os.path.exists(image_dir):
                image_dir = os.path.join(os.path.dirname(ann_dir), "images")

        all_samples = []
        all_labels = set()

        for fname in sorted(os.listdir(ann_dir)):
            if not fname.lower().endswith(".xml"):
                continue
            xml_path = os.path.join(ann_dir, fname)
            sample, labels = self._parse_voc_xml(xml_path, image_dir, load_images)
            if sample is not None:
                all_samples.append(sample)
                all_labels.update(labels)

        label_names = sorted(all_labels)
        has_bbox = any(ann.bbox is not None for s in all_samples for ann in s.annotations)
        task_type = "detection" if has_bbox else "classification"

        return DataObject(
            samples=all_samples,
            label_names=label_names,
            task_type=task_type,
            source_format="voc",
            metadata={
                "annotation_dir": ann_dir,
                "image_dir": image_dir,
                "num_samples": len(all_samples),
            },
        )

    def _load_single_annotation(self, xml_path: str, image_dir: Optional[str],
                                load_images: bool) -> DataObject:
        if image_dir is None:
            image_dir = os.path.dirname(xml_path)

        sample, labels = self._parse_voc_xml(xml_path, image_dir, load_images)
        if sample is None:
            return DataObject(source_format="voc")

        label_names = sorted(labels)
        has_bbox = any(ann.bbox is not None for ann in sample.annotations)
        task_type = "detection" if has_bbox else "classification"

        return DataObject(
            samples=[sample],
            label_names=label_names,
            task_type=task_type,
            source_format="voc",
            metadata={"source_file": xml_path},
        )

    def _parse_voc_xml(self, xml_path: str, image_dir: Optional[str],
                       load_images: bool) -> tuple:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            return None, set()

        filename_elem = root.find("filename")
        filename = filename_elem.text if filename is not None else os.path.basename(xml_path).replace(".xml", ".jpg")

        size_elem = root.find("size")
        img_width = 0
        img_height = 0
        if size_elem is not None:
            w_elem = size_elem.find("width")
            h_elem = size_elem.find("height")
            if w_elem is not None:
                img_width = int(w_elem.text)
            if h_elem is not None:
                img_height = int(h_elem.text)

        sample_id = os.path.splitext(filename)[0]
        annotations = []
        labels = set()

        for obj in root.findall("object"):
            name_elem = obj.find("name")
            label = name_elem.text if name_elem is not None else "unknown"
            labels.add(label)

            difficult = 0
            diff_elem = obj.find("difficult")
            if diff_elem is not None:
                try:
                    difficult = int(diff_elem.text)
                except ValueError:
                    pass

            bbox = None
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                try:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    bbox = BBox(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax)
                except (AttributeError, ValueError, TypeError):
                    pass

            annotations.append(Annotation(
                label=label,
                bbox=bbox,
                extra={"difficult": difficult},
            ))

        sample = Sample(
            sample_id=sample_id,
            file_path=filename,
            annotations=annotations,
            extra={"width": img_width, "height": img_height},
        )

        if load_images and image_dir:
            full_path = os.path.join(image_dir, filename)
            if os.path.exists(full_path):
                try:
                    with open(full_path, "rb") as f:
                        sample.data = f.read()
                except Exception:
                    pass

        return sample, labels

    def validate(self, path: str, **kwargs) -> ValidationResult:
        errors = []
        warnings = []

        if os.path.isdir(path):
            xml_files = [f for f in os.listdir(path) if f.lower().endswith(".xml")]
            if not xml_files:
                errors.append("目录中没有找到 XML 标注文件")
                return ValidationResult(False, errors=errors)

            for fname in xml_files[:50]:
                xml_path = os.path.join(path, fname)
                result = self._validate_single_xml(xml_path)
                errors.extend(result.errors)
                warnings.extend(result.warnings)

            return ValidationResult(len(errors) == 0, errors=errors, warnings=warnings)

        return self._validate_single_xml(path)

    def _validate_single_xml(self, xml_path: str) -> ValidationResult:
        errors = []
        warnings = []

        if not os.path.exists(xml_path):
            return ValidationResult(False, errors=[f"文件不存在: {xml_path}"])

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            return ValidationResult(False, errors=[f"XML 解析失败: {str(e)}"])

        if root.find("filename") is None:
            warnings.append("缺少 filename 字段")

        size_elem = root.find("size")
        if size_elem is None:
            warnings.append("缺少 size 字段（图像尺寸信息）")
        else:
            if size_elem.find("width") is None or size_elem.find("height") is None:
                warnings.append("size 中缺少 width 或 height 字段")

        for i, obj in enumerate(root.findall("object")):
            if obj.find("name") is None:
                errors.append(f"object[{i}] 缺少 name 字段（类别名称）")

            bndbox = obj.find("bndbox")
            if bndbox is not None:
                try:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)

                    if xmin < 0 or ymin < 0:
                        warnings.append(f"object[{i}] bbox 坐标出现负值")
                    if xmax <= xmin or ymax <= ymin:
                        errors.append(f"object[{i}] bbox 无效: xmax<=xmin 或 ymax<=ymin")
                except (AttributeError, ValueError, TypeError) as e:
                    errors.append(f"object[{i}] bbox 坐标解析失败: {str(e)}")

        return ValidationResult(len(errors) == 0, errors=errors, warnings=warnings)

    def get_metadata(self, path: str, **kwargs) -> MetadataInfo:
        if os.path.isdir(path):
            xml_files = [f for f in os.listdir(path) if f.lower().endswith(".xml")]
            all_labels = set()
            total_objects = 0

            for fname in xml_files:
                xml_path = os.path.join(path, fname)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        name_elem = obj.find("name")
                        if name_elem is not None:
                            all_labels.add(name_elem.text)
                        total_objects += 1
                except ET.ParseError:
                    continue

            label_names = sorted(all_labels)
            has_bbox = total_objects > 0
            task_type = "detection" if has_bbox else "classification"

            return MetadataInfo(
                num_classes=len(label_names),
                sample_count=len(xml_files),
                task_type=task_type,
                label_names=label_names,
                field_structure={"annotation": "filename, size, object(name, bndbox)"},
                extra={"total_objects": total_objects},
            )

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except ET.ParseError:
            return MetadataInfo()

        labels = set()
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is not None:
                labels.add(name_elem.text)

        label_names = sorted(labels)
        has_bbox = any(obj.find("bndbox") is not None for obj in root.findall("object"))

        return MetadataInfo(
            num_classes=len(label_names),
            sample_count=1,
            task_type="detection" if has_bbox else "classification",
            label_names=label_names,
            field_structure={"annotation": "filename, size, object(name, bndbox)"},
        )
