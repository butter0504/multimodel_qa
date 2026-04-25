import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULT_CONFIG = {
    'data': {
        'data_root': './data/',
        'dataset_name': None,
        'label_column': None,
        'file_encoding': 'utf-8',
    },
    'detection': {
        'method': 'confident_learning',
        'threshold': 0.3,
        'cross_validation_folds': 5,
        'percentile_threshold': 85,
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42,
        'blur_threshold': 50,
        'noise_threshold': 25.0,
        'min_resolution': 32,
        'exposure_ratio_threshold': 0.5,
        'contrast_threshold': 0.08,
        'ks_significance': 0.05,
        'js_threshold': 0.1,
        'k_neighbors': 5,
        'uncertainty_std_factor': 1.5,
    },
    'feature_extraction': {
        'image_model': 'resnet50',
        'text_model': 'tfidf',
        'tabular_method': 'standard_scaler',
        'batch_size': 32,
        'max_features_tfidf': 10000,
        'image_resize': 224,
    },
    'report': {
        'output_dir': './reports/',
        'include_visualization': True,
        'include_embedding': False,
        'include_suggestion': True,
        'format': 'json',
        'sample_output_count': 5,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """统一配置管理类，借鉴 Docta 的 Config 设计"""

    def __init__(self, cfg_dict: Optional[Dict[str, Any]] = None):
        self._cfg = _deep_merge(_DEFAULT_CONFIG, cfg_dict or {})
        self._apply_cfg(self._cfg)

    def _apply_cfg(self, cfg: dict):
        self.data = cfg.get('data', {})
        self.detection = cfg.get('detection', {})
        self.feature_extraction = cfg.get('feature_extraction', {})
        self.report = cfg.get('report', {})

    @classmethod
    def fromfile(cls, filepath: str) -> 'Config':
        if not os.path.exists(filepath):
            print(f"配置文件 {filepath} 不存在，使用默认配置")
            return cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f) or {}
        return cls(cfg_dict)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> 'Config':
        return cls(cfg_dict)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self._cfg
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any):
        keys = key_path.split('.')
        cfg = self._cfg
        for k in keys[:-1]:
            if k not in cfg or not isinstance(cfg[k], dict):
                cfg[k] = {}
            cfg = cfg[k]
        cfg[keys[-1]] = value
        self._apply_cfg(self._cfg)

    def to_dict(self) -> dict:
        import copy
        return copy.deepcopy(self._cfg)

    def __repr__(self):
        return f"Config({self._cfg})"
