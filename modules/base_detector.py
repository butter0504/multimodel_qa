from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from .config import Config
from .report import Report, DiagnosisSection, IssueRecord


class BaseDetector(ABC):
    """检测器基类，集成配置管理和报告生成"""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.report = Report(self.cfg)
        self.issues: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}

    @abstractmethod
    def detect(self, data: Any, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_issues(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def modality(self) -> str:
        pass

    def _init_report(self, dataset_info: Dict[str, Any]):
        self.report = Report(self.cfg)
        self.report.set_dataset_info({
            **dataset_info,
            'modality': self.modality(),
        })

    def _add_issue_to_report(self, section_name: str, record: IssueRecord):
        if section_name not in self.report.sections:
            self.report.add_section(DiagnosisSection(section_name))
        self.report.sections[section_name].add_issue(record)

    def _set_noise_rate(self, issue_type: str, found: int, total: int):
        self.report.set_noise_rate(issue_type, found, total)

    def _add_cured_sample(self, sample: Dict[str, Any]):
        self.report.add_cured_sample(sample)

    def get_report(self) -> Report:
        return self.report

    def generate_report_json(self, indent: int = 2) -> str:
        return self.report.to_json(indent=indent)

    def save_report(self, filepath: Optional[str] = None) -> str:
        return self.report.save(filepath)

    def print_report_summary(self):
        self.report.print_summary()
