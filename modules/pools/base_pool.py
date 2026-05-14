from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np

from ..config import Config


@dataclass
class MethodResult:
    method_name: str
    scores: np.ndarray
    execution_time: float = 0.0
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "scores": self.scores.tolist() if isinstance(self.scores, np.ndarray) else self.scores,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class PoolResult:
    pool_name: str
    method_results: List[MethodResult] = field(default_factory=list)
    ensemble_scores: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_name": self.pool_name,
            "method_results": [mr.to_dict() for mr in self.method_results],
            "ensemble_scores": self.ensemble_scores.tolist() if self.ensemble_scores is not None else None,
            "metadata": self.metadata,
        }


class BasePool(ABC):
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self._methods: Dict[str, callable] = {}
        self._register_methods()

    @abstractmethod
    def _register_methods(self):
        pass

    @abstractmethod
    def pool_name(self) -> str:
        pass

    def list_methods(self) -> List[str]:
        return list(self._methods.keys())

    def run_method(self, method_name: str, **kwargs) -> MethodResult:
        import time

        if method_name not in self._methods:
            return MethodResult(
                method_name=method_name,
                scores=np.array([]),
                success=False,
                error_message=f"方法 '{method_name}' 不存在于池中",
            )

        start = time.time()
        try:
            scores = self._methods[method_name](**kwargs)
            elapsed = time.time() - start
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores, dtype=np.float64)
            scores = np.clip(scores, 0.0, 1.0)
            return MethodResult(
                method_name=method_name,
                scores=scores,
                execution_time=elapsed,
                success=True,
            )
        except Exception as e:
            elapsed = time.time() - start
            return MethodResult(
                method_name=method_name,
                scores=np.array([]),
                execution_time=elapsed,
                success=False,
                error_message=str(e),
            )

    def run_all(self, **kwargs) -> PoolResult:
        import time

        start = time.time()
        results = []
        for name in self._methods:
            mr = self.run_method(name, **kwargs)
            results.append(mr)

        ensemble = self._ensemble(results)

        elapsed = time.time() - start
        return PoolResult(
            pool_name=self.pool_name(),
            method_results=results,
            ensemble_scores=ensemble,
            metadata={"total_execution_time": elapsed},
        )

    def _ensemble(self, results: List[MethodResult]) -> Optional[np.ndarray]:
        valid = [r for r in results if r.success and len(r.scores) > 0]
        if not valid:
            return None
        try:
            scores_stack = np.vstack([r.scores for r in valid])
            return np.mean(scores_stack, axis=0)
        except ValueError:
            return valid[0].scores
