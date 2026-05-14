from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

import numpy as np

from .base_pool import BasePool, MethodResult
from ..config import Config
from ..data_object import DataObject, Sample, Annotation, BBox


class FormatStructurePool(BasePool):
    """
    格式结构规则池
    ==============
    包含6条规则校验，采用快速失败策略，按计算成本从低到高执行。

    规则列表（按计算成本从低到高排序）：
        1. category_index_check - 类别索引越界检查
        2. coordinate_range_check - 坐标值域校验（边界框在图像内）
        3. path_validity_check   - 路径有效性检查（引用的文件存在）
        4. csv_consistency_check  - CSV列一致性（行列数匹配）
        5. json_schema_check     - JSON Schema校验（必填字段、类型匹配）
        6. xml_integrity_check   - XML标签完整性（闭合、关键节点）
    """

    def pool_name(self) -> str:
        return "format_structure"

    def _register_methods(self):
        self._methods = {
            "category_index_check": self.category_index_check,
            "coordinate_range_check": self.coordinate_range_check,
            "path_validity_check": self.path_validity_check,
            "csv_consistency_check": self.csv_consistency_check,
            "json_schema_check": self.json_schema_check,
            "xml_integrity_check": self.xml_integrity_check,
        }

    def run_all(self, **kwargs) -> Any:
        from .base_pool import PoolResult
        import time

        start = time.time()
        results = []
        failed = False

        for name in self._methods:
            mr = self.run_method(name, **kwargs)
            results.append(mr)
            if mr.success and len(mr.scores) > 0 and np.any(mr.scores > 0):
                failed = True
                break

        elapsed = time.time() - start
        return PoolResult(
            pool_name=self.pool_name(),
            method_results=results,
            ensemble_scores=None,
            metadata={
                "total_execution_time": elapsed,
                "fast_fail_triggered": failed,
                "execution_order": list(self._methods.keys()),
            },
        )

    def category_index_check(self, data_object: Optional[DataObject] = None,
                             labels: Optional[List] = None,
                             label_names: Optional[List[str]] = None,
                             **kwargs) -> np.ndarray:
        """
        类别索引越界检查。

        检查每个样本的标签是否在合法的类别列表中。
        若标签不在 label_names 中，则视为越界。

        Parameters
        ----------
        data_object : Optional[DataObject]
            数据对象，若提供则从中提取标签和类别名
        labels : Optional[List]
            标签列表（当 data_object 不可用时）
        label_names : Optional[List[str]]
            合法类别名列表

        Returns
        -------
        np.ndarray
            异常分数 (n_samples,)，0.0=合法，1.0=越界
        """
        if data_object is not None:
            sample_labels = data_object.get_labels()
            valid_names = set(data_object.label_names)
        else:
            sample_labels = labels or []
            valid_names = set(label_names) if label_names else set()

        if not valid_names:
            return np.zeros(len(sample_labels), dtype=np.float64)

        scores = []
        violations = []
        for i, label in enumerate(sample_labels):
            if label is None or str(label) not in valid_names:
                scores.append(1.0)
                violations.append({
                    "index": i,
                    "label": label,
                    "reason": f"标签 '{label}' 不在合法类别 {valid_names} 中",
                })
            else:
                scores.append(0.0)

        return np.array(scores, dtype=np.float64)

    def coordinate_range_check(self, data_object: Optional[DataObject] = None,
                               bboxes: Optional[List[Dict]] = None,
                               image_sizes: Optional[List[Dict]] = None,
                               **kwargs) -> np.ndarray:
        """
        坐标值域校验（边界框在图像内）。

        检查每个边界框的坐标是否在图像范围内。
        坐标超出图像边界视为异常。

        Parameters
        ----------
        data_object : Optional[DataObject]
            数据对象，若提供则从中提取边界框信息
        bboxes : Optional[List[Dict]]
            边界框列表，每个元素为 {"x_min", "y_min", "x_max", "y_max"}
        image_sizes : Optional[List[Dict]]
            图像尺寸列表，每个元素为 {"height", "width"}

        Returns
        -------
        np.ndarray
            异常分数 (n_samples,)，0.0=合法，1.0=越界
        """
        if data_object is not None:
            n_samples = data_object.sample_count
            bbox_list = []
            size_list = []
            for s in data_object.samples:
                if s.annotations and s.annotations[0].bbox is not None:
                    bbox = s.annotations[0].bbox
                    bbox_list.append({
                        "x_min": bbox.x_min, "y_min": bbox.y_min,
                        "x_max": bbox.x_max, "y_max": bbox.y_max,
                    })
                else:
                    bbox_list.append(None)

                extra = s.extra
                if "height" in extra and "width" in extra:
                    size_list.append({"height": extra["height"], "width": extra["width"]})
                else:
                    size_list.append(None)
        else:
            bbox_list = bboxes or []
            size_list = image_sizes or []
            n_samples = len(bbox_list)

        scores = []
        violations = []
        for i in range(n_samples):
            bbox = bbox_list[i] if i < len(bbox_list) else None
            size = size_list[i] if i < len(size_list) else None

            if bbox is None:
                scores.append(0.0)
                continue

            x_min = bbox.get("x_min", 0)
            y_min = bbox.get("y_min", 0)
            x_max = bbox.get("x_max", 0)
            y_max = bbox.get("y_max", 0)

            is_invalid = False
            reasons = []

            if x_min < 0 or y_min < 0:
                is_invalid = True
                reasons.append(f"坐标 ({x_min}, {y_min}) 小于0")

            if x_max <= x_min or y_max <= y_min:
                is_invalid = True
                reasons.append(f"边界框无效: ({x_min},{y_min})-({x_max},{y_max})")

            if size is not None:
                h, w = size.get("height", float('inf')), size.get("width", float('inf'))
                if x_max > w or y_max > h:
                    is_invalid = True
                    reasons.append(f"坐标 ({x_max}, {y_max}) 超出图像范围 ({w}, {h})")

            if is_invalid:
                scores.append(1.0)
                violations.append({"index": i, "bbox": bbox, "reasons": reasons})
            else:
                scores.append(0.0)

        return np.array(scores, dtype=np.float64)

    def path_validity_check(self, data_object: Optional[DataObject] = None,
                            file_paths: Optional[List[str]] = None,
                            data_root: str = "", **kwargs) -> np.ndarray:
        """
        路径有效性检查（引用的文件存在）。

        检查每个样本引用的文件路径是否存在。

        Parameters
        ----------
        data_object : Optional[DataObject]
            数据对象
        file_paths : Optional[List[str]]
            文件路径列表
        data_root : str
            数据根目录，路径将相对于此目录解析

        Returns
        -------
        np.ndarray
            异常分数 (n_samples,)，0.0=存在，1.0=不存在
        """
        if data_object is not None:
            paths = [s.file_path for s in data_object.samples]
        else:
            paths = file_paths or []

        scores = []
        violations = []
        for i, path in enumerate(paths):
            if path is None:
                scores.append(0.0)
                continue

            full_path = os.path.join(data_root, path) if data_root else path
            if not os.path.exists(full_path):
                scores.append(1.0)
                violations.append({
                    "index": i,
                    "path": path,
                    "full_path": full_path,
                    "reason": f"文件不存在: {full_path}",
                })
            else:
                scores.append(0.0)

        return np.array(scores, dtype=np.float64)

    def csv_consistency_check(self, csv_data: Optional[str] = None,
                              csv_path: Optional[str] = None,
                              expected_columns: Optional[List[str]] = None,
                              **kwargs) -> np.ndarray:
        """
        CSV列一致性（行列数匹配）。

        检查CSV文件的列数是否一致，行数是否与预期匹配。

        Parameters
        ----------
        csv_data : Optional[str]
            CSV原始字符串数据
        csv_path : Optional[str]
            CSV文件路径
        expected_columns : Optional[List[str]]
            期望的列名列表

        Returns
        -------
        np.ndarray
            异常分数，形状 (1,)，0.0=一致，1.0=不一致
        """
        import csv

        if csv_path is not None:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
        elif csv_data is not None:
            reader = csv.reader(csv_data.splitlines())
            rows = list(reader)
        else:
            return np.array([0.0], dtype=np.float64)

        if not rows:
            return np.array([1.0], dtype=np.float64)

        n_cols = len(rows[0])
        inconsistent_rows = 0
        for i, row in enumerate(rows):
            if len(row) != n_cols:
                inconsistent_rows += 1

        if expected_columns is not None:
            header = rows[0] if rows else []
            missing = set(expected_columns) - set(header)
            if missing:
                return np.array([1.0], dtype=np.float64)

        anomaly = float(inconsistent_rows / len(rows)) if rows else 1.0
        return np.array([anomaly], dtype=np.float64)

    def json_schema_check(self, json_data: Optional[Dict] = None,
                          json_path: Optional[str] = None,
                          schema: Optional[Dict] = None,
                          required_fields: Optional[List[str]] = None,
                          **kwargs) -> np.ndarray:
        """
        JSON Schema校验（必填字段、类型匹配）。

        检查JSON数据是否符合给定的Schema，
        包括必填字段和类型匹配。

        Parameters
        ----------
        json_data : Optional[Dict]
            JSON数据字典
        json_path : Optional[str]
            JSON文件路径
        schema : Optional[Dict]
            JSON Schema字典
        required_fields : Optional[List[str]]
            必填字段列表（简单模式，无需完整Schema）

        Returns
        -------
        np.ndarray
            异常分数，形状 (1,)，0.0=合法，1.0=不合法
        """
        if json_path is not None:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

        if json_data is None:
            return np.array([0.0], dtype=np.float64)

        if schema is not None:
            try:
                import jsonschema
                jsonschema.validate(instance=json_data, schema=schema)
                return np.array([0.0], dtype=np.float64)
            except jsonschema.ValidationError as e:
                return np.array([1.0], dtype=np.float64)
            except ImportError:
                pass

        if required_fields is not None:
            missing = [f for f in required_fields if f not in json_data]
            if missing:
                return np.array([1.0], dtype=np.float64)

        if isinstance(json_data, list):
            violations = 0
            for item in json_data:
                if required_fields:
                    missing = [f for f in required_fields if f not in item]
                    if missing:
                        violations += 1
            if violations > 0:
                anomaly = float(violations / len(json_data))
                return np.array([anomaly], dtype=np.float64)

        return np.array([0.0], dtype=np.float64)

    def xml_integrity_check(self, xml_data: Optional[str] = None,
                            xml_path: Optional[str] = None,
                            required_tags: Optional[List[str]] = None,
                            **kwargs) -> np.ndarray:
        """
        XML标签完整性（闭合、关键节点）。

        检查XML文件是否格式正确（标签闭合），
        以及是否包含所有必需的关键节点。

        Parameters
        ----------
        xml_data : Optional[str]
            XML字符串数据
        xml_path : Optional[str]
            XML文件路径
        required_tags : Optional[List[str]]
            必需的标签名列表

        Returns
        -------
        np.ndarray
            异常分数，形状 (1,)，0.0=合法，1.0=不合法
        """
        if xml_path is not None:
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_data = f.read()

        if xml_data is None:
            return np.array([0.0], dtype=np.float64)

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            return np.array([1.0], dtype=np.float64)

        if required_tags is not None:
            found_tags = set()
            for elem in root.iter():
                found_tags.add(elem.tag)

            missing = [t for t in required_tags if t not in found_tags]
            if missing:
                anomaly = len(missing) / len(required_tags)
                return np.array([float(anomaly)], dtype=np.float64)

        return np.array([0.0], dtype=np.float64)
