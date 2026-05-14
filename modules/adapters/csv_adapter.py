from __future__ import annotations

import os
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional

from ..data_object import Annotation, DataObject, Sample
from .base_adapter import BaseAdapter, MetadataInfo, ValidationResult


class CSVAdapter(BaseAdapter):
    format_name = "csv"
    supported_extensions = [".csv", ".tsv"]

    def load(self, path: str, **kwargs) -> DataObject:
        import pandas as pd

        label_column = kwargs.get("label_column", None)
        text_column = kwargs.get("text_column", None)
        filename_column = kwargs.get("filename_column", None)
        encoding = kwargs.get("encoding", "utf-8")
        separator = kwargs.get("separator", None)

        if separator is None:
            ext = os.path.splitext(path)[1].lower()
            separator = "\t" if ext == ".tsv" else ","

        df = pd.read_csv(path, sep=separator, encoding=encoding)

        if label_column is None:
            label_column = self._guess_label_column(df)

        if filename_column is None:
            filename_column = self._guess_filename_column(df)

        if text_column is None:
            text_column = self._guess_text_column(df)

        label_names = []
        if label_column and label_column in df.columns:
            label_names = sorted(df[label_column].dropna().unique().tolist())

        samples = []
        for idx, row in df.iterrows():
            sample_id = str(idx)
            file_path = None
            data = None

            if filename_column and filename_column in df.columns:
                file_path = str(row[filename_column])

            if text_column and text_column in df.columns:
                text_content = str(row[text_column])
                data = text_content.encode("utf-8")

            annotations = []
            if label_column and label_column in df.columns and pd.notna(row[label_column]):
                annotations.append(Annotation(label=str(row[label_column])))

            extra = {}
            for col in df.columns:
                if col not in [label_column, filename_column, text_column]:
                    val = row[col]
                    if pd.notna(val):
                        try:
                            extra[col] = float(val) if isinstance(val, (int, float)) else str(val)
                        except (ValueError, TypeError):
                            extra[col] = str(val)

            samples.append(Sample(
                sample_id=sample_id,
                data=data,
                file_path=file_path,
                annotations=annotations,
                extra=extra,
            ))

        has_labels = label_column is not None and label_column in df.columns
        task_type = "classification" if has_labels else "unknown"

        return DataObject(
            samples=samples,
            label_names=label_names,
            task_type=task_type,
            source_format="csv",
            metadata={
                "source_file": path,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "label_column": label_column,
                "text_column": text_column,
                "filename_column": filename_column,
            },
        )

    def validate(self, path: str, **kwargs) -> ValidationResult:
        import pandas as pd

        errors = []
        warnings = []

        if not os.path.exists(path):
            return ValidationResult(False, errors=[f"文件不存在: {path}"])

        encoding = kwargs.get("encoding", "utf-8")
        separator = kwargs.get("separator", None)

        if separator is None:
            ext = os.path.splitext(path)[1].lower()
            separator = "\t" if ext == ".tsv" else ","

        try:
            df = pd.read_csv(path, sep=separator, encoding=encoding)
        except pd.errors.EmptyDataError:
            return ValidationResult(False, errors=["CSV 文件为空"])
        except pd.errors.ParserError as e:
            return ValidationResult(False, errors=[f"CSV 解析失败: {str(e)}"])
        except UnicodeDecodeError:
            return ValidationResult(False, errors=[f"编码错误，尝试使用 {encoding} 解码失败"])

        if len(df) == 0:
            warnings.append("CSV 文件没有数据行")

        if len(df.columns) == 1 and separator == ",":
            try:
                df_semicolon = pd.read_csv(path, sep=";", encoding=encoding)
                if len(df_semicolon.columns) > 1:
                    warnings.append("检测到可能使用了分号作为分隔符，但当前使用逗号解析")
            except Exception:
                pass

        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_ratio = null_count / len(df)
                if null_ratio > 0.5:
                    warnings.append(f"列 '{col}' 有 {null_ratio:.1%} 的缺失值")

        label_column = kwargs.get("label_column", self._guess_label_column(df))
        if label_column and label_column in df.columns:
            class_counts = df[label_column].value_counts()
            min_count = class_counts.min()
            max_count = class_counts.max()
            if min_count < 2:
                warnings.append(f"标签列 '{label_column}' 中某些类别样本数不足 2")
            if max_count / (min_count + 1) > 10:
                warnings.append(f"标签列 '{label_column}' 类别分布严重不均衡")

        return ValidationResult(len(errors) == 0, errors=errors, warnings=warnings)

    def get_metadata(self, path: str, **kwargs) -> DataObject:
        import pandas as pd

        encoding = kwargs.get("encoding", "utf-8")
        separator = kwargs.get("separator", None)

        if separator is None:
            ext = os.path.splitext(path)[1].lower()
            separator = "\t" if ext == ".tsv" else ","

        try:
            df = pd.read_csv(path, sep=separator, encoding=encoding)
        except Exception:
            return MetadataInfo()

        label_column = kwargs.get("label_column", self._guess_label_column(df))
        label_names = []
        if label_column and label_column in df.columns:
            label_names = sorted(df[label_column].dropna().unique().tolist())

        field_structure = {}
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            field_structure[col] = dtype_str

        has_labels = label_column is not None and label_column in df.columns

        return MetadataInfo(
            num_classes=len(label_names),
            sample_count=len(df),
            task_type="classification" if has_labels else "unknown",
            label_names=label_names,
            field_structure=field_structure,
            extra={
                "num_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "label_column": label_column,
            },
        )

    def _guess_label_column(self, df) -> Optional[str]:
        candidates = ["label", "class", "category", "target", "y", "标签", "类别"]
        for col in df.columns:
            if col.lower() in candidates:
                return col
        for col in df.columns:
            if col.lower() in [c + "s" for c in candidates]:
                return col
        return None

    def _guess_filename_column(self, df) -> Optional[str]:
        candidates = ["filename", "file_name", "image", "img", "path", "文件名"]
        for col in df.columns:
            if col.lower() in candidates:
                return col
        return None

    def _guess_text_column(self, df) -> Optional[str]:
        candidates = ["text", "content", "sentence", "review", "description", "文本", "内容"]
        for col in df.columns:
            if col.lower() in candidates:
                return col
        return None
