from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from .config import Config
from .data_object import DataObject
from .image_detector import ImageDetector
from .table_detector import TableDetector
from .text_detector import TextDetector


class DetectionTask:
    def __init__(self, task_id: str, modality: str, modules: List[str],
                 data_obj: DataObject, extra: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.modality = modality
        self.modules = modules
        self.data_obj = data_obj
        self.extra = extra or {}
        self.status: str = "pending"
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None


class Scheduler:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self._detectors: Dict[str, Any] = {}
        self._progress_callbacks: Dict[str, Callable] = {}
        self._init_detectors()

    def _init_detectors(self):
        self._detectors = {
            "image": ImageDetector(self.cfg),
            "text": TextDetector(self.cfg),
            "table": TableDetector(self.cfg),
        }

    def submit(self, task: DetectionTask,
               progress_callback: Optional[Callable] = None) -> DetectionTask:
        if progress_callback is not None:
            self._progress_callbacks[task.task_id] = progress_callback

        detector = self._detectors.get(task.modality)
        if detector is None:
            task.status = "failed"
            task.error = f"不支持的模态类型: {task.modality}"
            return task

        try:
            task.status = "running"
            result = self._dispatch(detector, task)
            task.result = result
            task.status = "completed"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)

        self._progress_callbacks.pop(task.task_id, None)
        return task

    def submit_batch(self, tasks: List[DetectionTask],
                     max_workers: int = 2,
                     progress_callback: Optional[Callable] = None) -> List[DetectionTask]:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task in tasks:
                future = executor.submit(self.submit, task, progress_callback)
                future_to_task[future] = task

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                results.append(task)

        return results

    def _dispatch(self, detector: Any, task: DetectionTask) -> Dict[str, Any]:
        data_obj = task.data_obj
        cb = self._progress_callbacks.get(task.task_id)

        if task.modality == "image":
            images = data_obj.get_images()
            labels = data_obj.get_labels()
            label_indices = data_obj.get_label_indices()

            use_labels = labels if any(l is not None for l in labels) else None
            if use_labels is None and any(i >= 0 for i in label_indices):
                use_labels = label_indices

            return detector.detect(
                data=images if images else [],
                labels=use_labels,
                modules=task.modules,
                progress_callback=cb,
            )

        elif task.modality == "text":
            texts = []
            for s in data_obj.samples:
                if s.data is not None:
                    texts.append(s.data.decode("utf-8", errors="replace"))
                elif s.file_path:
                    texts.append(s.file_path)
                else:
                    texts.append("")

            labels = data_obj.get_labels()
            use_labels = labels if any(l is not None for l in labels) else None

            return detector.detect(
                data=texts,
                labels=use_labels,
                modules=task.modules,
                progress_callback=cb,
            )

        elif task.modality == "table":
            import pandas as pd

            rows = []
            for s in data_obj.samples:
                row = dict(s.extra)
                if s.primary_label is not None:
                    row["label"] = s.primary_label
                rows.append(row)

            df = pd.DataFrame(rows) if rows else pd.DataFrame()
            label_col = task.extra.get("label_column", None)

            return detector.detect(
                data=df,
                label_col=label_col,
            )

        return {"error": f"未知的模态类型: {task.modality}"}

    def create_task(self, data_obj: DataObject, modules: Optional[List[str]] = None,
                    task_id: Optional[str] = None,
                    extra: Optional[Dict[str, Any]] = None) -> DetectionTask:
        import uuid

        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        modality = self._infer_modality(data_obj)
        if modules is None:
            modules = self._default_modules(modality)

        return DetectionTask(
            task_id=task_id,
            modality=modality,
            modules=modules,
            data_obj=data_obj,
            extra=extra,
        )

    def _infer_modality(self, data_obj: DataObject) -> str:
        if data_obj.metadata.get("modality"):
            return data_obj.metadata["modality"]

        if data_obj.task_type == "detection":
            return "image"

        has_images = any(s.data is not None for s in data_obj.samples)
        if has_images:
            return "image"

        if data_obj.metadata.get("columns"):
            return "table"

        for s in data_obj.samples:
            if s.data is not None:
                try:
                    text = s.data.decode("utf-8")
                    if len(text) > 20 and not text.startswith(("\x89PNG", "\xff\xd8\xff")):
                        return "text"
                except (UnicodeDecodeError, ValueError):
                    return "image"

        return "table"

    def _default_modules(self, modality: str) -> List[str]:
        defaults = {
            "image": ["basic_quality"],
            "text": ["text_length", "duplicate", "character_anomaly"],
            "table": ["missing_values", "outliers", "duplicates"],
        }
        return defaults.get(modality, [])

    def aggregate_results(self, tasks: List[DetectionTask]) -> Dict[str, Any]:
        all_issues = []
        all_metrics = {}
        module_results = {}
        errors = []

        for task in tasks:
            if task.status == "failed":
                errors.append({"task_id": task.task_id, "error": task.error})
                continue

            if task.result is None:
                continue

            issues = task.result.get("issues", [])
            all_issues.extend(issues)

            metrics = task.result.get("metrics", {})
            all_metrics[task.task_id] = metrics

            for key, value in task.result.items():
                if key not in ("issues", "metrics", "diagnosis_report"):
                    module_results.setdefault(task.modality, {})[key] = value

        total_samples = sum(
            t.data_obj.sample_count for t in tasks if t.data_obj
        )

        return {
            "total_tasks": len(tasks),
            "completed_tasks": sum(1 for t in tasks if t.status == "completed"),
            "failed_tasks": sum(1 for t in tasks if t.status == "failed"),
            "total_samples": total_samples,
            "total_issues": len(all_issues),
            "issues": all_issues,
            "metrics": all_metrics,
            "module_results": module_results,
            "errors": errors,
        }
