import json
import os
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import Config


class IssueRecord:
    def __init__(self, index: int, issue_type: str, original_label: Any = None,
                 suggested_label: Any = None, quality_score: float = None,
                 details: Optional[Dict[str, Any]] = None):
        self.index = index
        self.issue_type = issue_type
        self.original_label = original_label
        self.suggested_label = suggested_label
        self.quality_score = quality_score
        self.details = details or {}

    def to_dict(self) -> dict:
        d = {
            'index': self.index,
            'issue_type': self.issue_type,
        }
        if self.original_label is not None:
            d['original_label'] = _serialize(self.original_label)
        if self.suggested_label is not None:
            d['suggested_label'] = _serialize(self.suggested_label)
        if self.quality_score is not None:
            d['quality_score'] = _serialize(self.quality_score)
        if self.details:
            d['details'] = _serialize(self.details)
        return d


class DiagnosisSection:
    def __init__(self, name: str, description: str = ''):
        self.name = name
        self.description = description
        self.metrics: Dict[str, Any] = {}
        self.issues: List[IssueRecord] = []
        self.statistics: Dict[str, Any] = {}

    def add_metric(self, key: str, value: Any):
        self.metrics[key] = value

    def add_issue(self, issue: IssueRecord):
        self.issues.append(issue)

    def add_statistic(self, key: str, value: Any):
        self.statistics[key] = value

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'metrics': _serialize(self.metrics),
            'statistics': _serialize(self.statistics),
            'issues': [issue.to_dict() for issue in self.issues],
            'issue_count': len(self.issues),
        }


class Report:
    """Docta 风格的诊断报告，统一管理所有检测结果"""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.created_at = datetime.now().isoformat()
        self.dataset_info: Dict[str, Any] = {}
        self.sections: Dict[str, DiagnosisSection] = {}
        self.noise_rates: Dict[str, str] = {}
        self.summary: Dict[str, Any] = {}
        self.cured_samples: List[Dict[str, Any]] = []
        self.config_snapshot: Dict[str, Any] = self.cfg.to_dict()

    def set_dataset_info(self, info: Dict[str, Any]):
        self.dataset_info = info

    def add_section(self, section: DiagnosisSection):
        self.sections[section.name] = section

    def get_section(self, name: str) -> Optional[DiagnosisSection]:
        return self.sections.get(name)

    def set_noise_rate(self, issue_type: str, found: int, total: int):
        if total > 0:
            rate = found / total
            self.noise_rates[issue_type] = f"{found}/{total} ≈ {rate:.1%}"
        else:
            self.noise_rates[issue_type] = f"0/{total} ≈ 0%"

    def add_cured_sample(self, sample: Dict[str, Any]):
        self.cured_samples.append(sample)

    def build_summary(self):
        total_issues = 0
        issue_breakdown = {}
        for name, section in self.sections.items():
            count = len(section.issues)
            total_issues += count
            issue_breakdown[name] = count

        total_samples = self.dataset_info.get('total_samples', 0)
        overall_noise_rate = total_issues / total_samples if total_samples > 0 else 0

        self.summary = {
            'total_samples': total_samples,
            'total_issues': total_issues,
            'overall_noise_rate': f"{overall_noise_rate:.1%}",
            'issue_breakdown': issue_breakdown,
            'noise_rates': self.noise_rates,
            'modality': self.dataset_info.get('modality', 'unknown'),
            'detection_method': self.cfg.get('detection.method', 'confident_learning'),
        }

    def to_dict(self) -> dict:
        self.build_summary()
        return {
            'report_metadata': {
                'created_at': self.created_at,
                'system': '基于置信学习的多模态数据质量检测系统',
                'version': '1.0.0',
                'config': _serialize(self.config_snapshot),
            },
            'dataset_info': _serialize(self.dataset_info),
            'diagnosis_summary': _serialize(self.summary),
            'noise_rates': _serialize(self.noise_rates),
            'diagnosis_sections': {
                name: section.to_dict()
                for name, section in self.sections.items()
            },
            'cured_samples': _serialize(self.cured_samples),
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii, default=_json_default)

    def save(self, filepath: Optional[str] = None):
        output_dir = self.cfg.get('report.output_dir', './reports/')
        os.makedirs(output_dir, exist_ok=True)

        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            modality = self.dataset_info.get('modality', 'unknown')
            fmt = self.cfg.get('report.format', 'json')
            filepath = os.path.join(output_dir, f"diagnosis_report_{modality}_{timestamp}.{fmt}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

        return filepath

    def print_summary(self):
        self.build_summary()
        print("=" * 60)
        print("  数据质量诊断报告 (Diagnosis Report)")
        print("=" * 60)
        print(f"  系统: 基于置信学习的多模态数据质量检测系统")
        print(f"  时间: {self.created_at}")
        print(f"  检测方法: {self.cfg.get('detection.method', 'confident_learning')}")
        print("-" * 60)
        print(f"  数据集: {self.dataset_info.get('name', 'N/A')}")
        print(f"  模态类型: {self.dataset_info.get('modality', 'N/A')}")
        print(f"  样本总数: {self.dataset_info.get('total_samples', 0)}")
        print("-" * 60)
        print("  噪声率 (Noise Rates):")
        for issue_type, rate_str in self.noise_rates.items():
            print(f"    {issue_type}: {rate_str}")
        print("-" * 60)
        print("  诊断概要:")
        print(f"    总问题数: {self.summary.get('total_issues', 0)}")
        print(f"    总体噪声率: {self.summary.get('overall_noise_rate', '0%')}")
        for name, count in self.summary.get('issue_breakdown', {}).items():
            print(f"    {name}: {count} 个问题")
        print("=" * 60)

        if self.cured_samples:
            sample_count = self.cfg.get('report.sample_output_count', 5)
            print(f"\n  修复样本示例 (前 {sample_count} 个):")
            for i, sample in enumerate(self.cured_samples[:sample_count]):
                print(f"\n  --- 样本 {i + 1} ---")
                for key, value in sample.items():
                    if key == 'original_data':
                        val_str = str(value)
                        if len(val_str) > 100:
                            val_str = val_str[:100] + "..."
                        print(f"    原始数据: {val_str}")
                    elif key == 'suggested_label':
                        print(f"    建议标签: {value}")
                    elif key == 'original_label':
                        print(f"    原始标签: {value}")
                    elif key == 'quality_score':
                        print(f"    质量分数: {value}")
                    else:
                        print(f"    {key}: {value}")
            print("=" * 60)


def _serialize(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
