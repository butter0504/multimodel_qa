from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ..data_object import Annotation, BBox, DataObject, Sample
from .base_adapter import BaseAdapter, MetadataInfo, ValidationResult
from dotenv import load_dotenv
load_dotenv()

_ANNOTATION_SCHEMA = {
    "type": "object",
    "required": ["samples"],
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["sample_id"],
                "properties": {
                    "sample_id": {"type": "string"},
                    "label": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "extra": {"type": "object"},
                },
            },
        },
        "label_names": {
            "type": "array",
            "items": {"type": "string"},
        },
        "task_type": {"type": "string"},
    },
}

_PROMPT_TEMPLATE = """你是一个数据格式解析专家。请将下面的原始数据解析为统一的 JSON 结构。

## 任务说明
将输入的原始标注数据转换为以下标准格式。你需要识别其中的样本标识、类别标签和边界框坐标（如果存在）。

## 标准 Schema 定义
输出必须严格遵循以下 JSON 结构：
```json
{{
  "samples": [
    {{
      "sample_id": "样本唯一标识（字符串）",
      "label": "类别标签",
      "bbox": [x_min, y_min, x_max, y_max],
      "extra": {{}}
    }}
  ],
  "label_names": ["所有出现的类别名称列表"],
  "task_type": "classification 或 detection"
}}
```

注意：
- bbox 字段仅在数据中包含边界框坐标时填写，否则省略
- bbox 格式为 [x_min, y_min, x_max, y_max]，坐标为绝对像素值
- 如果坐标为归一化值（0-1），请在 extra 中标注 "normalized": true
- sample_id 必须是字符串类型

## Few-shot 示例

### 示例 1：YOLO 风格输入
输入：
```
0 0.5 0.3 0.2 0.4
1 0.3 0.6 0.15 0.3
```
类别映射：0=cat, 1=dog

输出：
```json
{{
  "samples": [
    {{
      "sample_id": "0",
      "label": "cat",
      "bbox": [0.4, 0.1, 0.6, 0.5],
      "extra": {{"normalized": true, "class_id": 0}}
    }},
    {{
      "sample_id": "1",
      "label": "dog",
      "bbox": [0.225, 0.45, 0.375, 0.75],
      "extra": {{"normalized": true, "class_id": 1}}
    }}
  ],
  "label_names": ["cat", "dog"],
  "task_type": "detection"
}}
```

### 示例 2：简单文本标注输入
输入：
```
img001 cat 120 80 300 250
img002 dog 50 30 200 180
img003 cat
```

输出：
```json
{{
  "samples": [
    {{
      "sample_id": "img001",
      "label": "cat",
      "bbox": [120, 80, 300, 250],
      "extra": {{}}
    }},
    {{
      "sample_id": "img002",
      "label": "dog",
      "bbox": [50, 30, 200, 180],
      "extra": {{}}
    }},
    {{
      "sample_id": "img003",
      "label": "cat",
      "extra": {{}}
    }}
  ],
  "label_names": ["cat", "dog"],
  "task_type": "detection"
}}
```

### 示例 3：纯分类标签输入
输入：
```
sample_01 positive
sample_02 negative
sample_03 positive
```

输出：
```json
{{
  "samples": [
    {{"sample_id": "sample_01", "label": "positive"}},
    {{"sample_id": "sample_02", "label": "negative"}},
    {{"sample_id": "sample_03", "label": "positive"}}
  ],
  "label_names": ["negative", "positive"],
  "task_type": "classification"
}}
```

## 当前待解析数据
```
{raw_data}
```

请严格按照上述 Schema 输出 JSON，不要添加任何额外说明文字。"""


class LLMAdapter(BaseAdapter):
    format_name = "llm"
    supported_extensions = [".txt", ".log", ".dat", ".csv", ".json", ".xml"]

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._llm_client = None

    def load(self, path: str, **kwargs) -> DataObject:
        raw_data = kwargs.get("raw_data", None)
        if raw_data is None:
            if not os.path.exists(path):
                return DataObject(source_format="llm", metadata={"error": "文件不存在"})
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    raw_data = f.read()
            except Exception as e:
                return DataObject(source_format="llm", metadata={"error": str(e)})

        if len(raw_data.strip()) == 0:
            return DataObject(source_format="llm", metadata={"error": "数据为空"})

        parsed = self._call_llm(raw_data)

        if parsed is None:
            data_obj = DataObject(
                source_format="llm",
                metadata={"parse_status": "failed", "needs_human_review": True},
            )
            data_obj.flag_sample("all", "LLM 解析失败，需人工介入")
            return data_obj

        schema_valid = self._validate_schema(parsed)
        if not schema_valid:
            data_obj = DataObject(
                source_format="llm",
                metadata={"parse_status": "schema_validation_failed", "needs_human_review": True},
            )
            data_obj.flag_sample("all", "LLM 输出 Schema 校验失败，需人工介入")
            return data_obj

        return self._build_data_object(parsed)

    def _call_llm(self, raw_data: str) -> Optional[Dict[str, Any]]:
        prompt = _PROMPT_TEMPLATE.format(raw_data=raw_data[:3000])

        try:
            client = self._get_llm_client()
            if client is not None:
                response = client.chat.completions.create(
                    model=self._get_model_name(),
                    messages=[
                        {"role": "system", "content": "你是一个数据格式解析专家，只输出 JSON，不添加任何额外说明。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content.strip()
                return self._extract_json(content)
        except Exception:
            pass

        return self._rule_based_fallback(raw_data)

    def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client

        try:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", None)

            if not api_key:
                return None

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url

            self._llm_client = OpenAI(**kwargs)
            return self._llm_client
        except ImportError:
            return None

    def _get_model_name(self) -> str:
        if self.cfg:
            return self.cfg.get("adapter.llm_model", "gpt-3.5-turbo")
        return os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            text = json_match.group(1).strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        brace_count = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0 and start >= 0:
                    try:
                        result = json.loads(text[start:i + 1])
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        start = -1

        return None

    def _validate_schema(self, parsed: Dict[str, Any]) -> bool:
        if "samples" not in parsed:
            return False
        if not isinstance(parsed["samples"], list):
            return False

        for sample in parsed["samples"]:
            if not isinstance(sample, dict):
                return False
            if "sample_id" not in sample:
                return False
            if "bbox" in sample:
                bbox = sample["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    return False
                if not all(isinstance(v, (int, float)) for v in bbox):
                    return False

        return True

    def _rule_based_fallback(self, raw_data: str) -> Optional[Dict[str, Any]]:
        lines = [l.strip() for l in raw_data.strip().splitlines() if l.strip()]
        if not lines:
            return None

        samples = []
        all_labels = set()
        has_bbox = False

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    nums = [float(p) for p in parts[-4:]]
                    sample_id = parts[0] if not parts[0].replace(".", "").isdigit() else str(i)
                    label = parts[1] if len(parts) > 1 else "unknown"
                    bbox = nums
                    has_bbox = True
                except ValueError:
                    sample_id = parts[0]
                    label = parts[1] if len(parts) > 1 else "unknown"
                    bbox = None

                sample = {"sample_id": sample_id, "label": label}
                if bbox:
                    sample["bbox"] = bbox
                samples.append(sample)
                all_labels.add(label)

            elif len(parts) == 2:
                sample_id = parts[0]
                label = parts[1]
                samples.append({"sample_id": sample_id, "label": label})
                all_labels.add(label)

            elif len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x_min = cx - w / 2
                    y_min = cy - h / 2
                    x_max = cx + w / 2
                    y_max = cy + h / 2
                    samples.append({
                        "sample_id": str(i),
                        "label": str(class_id),
                        "bbox": [x_min, y_min, x_max, y_max],
                        "extra": {"normalized": True, "class_id": class_id},
                    })
                    has_bbox = True
                except ValueError:
                    samples.append({"sample_id": str(i), "label": line})
                    all_labels.add(line)

            elif len(parts) == 1:
                samples.append({"sample_id": str(i), "label": parts[0]})
                all_labels.add(parts[0])

        if not samples:
            return None

        return {
            "samples": samples,
            "label_names": sorted(all_labels),
            "task_type": "detection" if has_bbox else "classification",
        }

    def _build_data_object(self, parsed: Dict[str, Any]) -> DataObject:
        label_names = parsed.get("label_names", [])
        task_type = parsed.get("task_type", "classification")

        samples = []
        for item in parsed.get("samples", []):
            sample_id = str(item.get("sample_id", ""))
            label = item.get("label", "")

            bbox = None
            if "bbox" in item and item["bbox"] is not None:
                coords = item["bbox"]
                if len(coords) == 4:
                    bbox = BBox(
                        x_min=float(coords[0]),
                        y_min=float(coords[1]),
                        x_max=float(coords[2]),
                        y_max=float(coords[3]),
                    )

            annotations = []
            if label:
                annotations.append(Annotation(label=label, bbox=bbox, extra=item.get("extra", {})))

            samples.append(Sample(
                sample_id=sample_id,
                annotations=annotations,
                extra=item.get("extra", {}),
            ))

        if not label_names:
            label_names = sorted(set(
                ann.label for s in samples for ann in s.annotations if ann.label
            ))

        return DataObject(
            samples=samples,
            label_names=label_names,
            task_type=task_type,
            source_format="llm",
            metadata={"parse_status": "success", "needs_human_review": False},
        )

    def validate(self, path: str, **kwargs) -> ValidationResult:
        if not os.path.exists(path):
            return ValidationResult(False, errors=[f"文件不存在: {path}"])

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return ValidationResult(False, errors=[f"文件读取失败: {str(e)}"])

        if len(content.strip()) == 0:
            return ValidationResult(False, errors=["文件内容为空"])

        warnings = []
        lines = content.strip().splitlines()
        if len(lines) > 10000:
            warnings.append("文件行数超过 10000，LLM 解析可能较慢")

        return ValidationResult(True, warnings=warnings)

    def get_metadata(self, path: str, **kwargs) -> MetadataInfo:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return MetadataInfo()

        lines = [l for l in content.strip().splitlines() if l.strip()]
        return MetadataInfo(
            sample_count=len(lines),
            task_type="unknown",
            extra={"file_size": len(content), "line_count": len(lines)},
        )
