import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from .base_detector import BaseDetector
from .cleanlab_wrapper import CleanlabWrapper
from .config import Config
from .report import DiagnosisSection, IssueRecord
from typing import Dict, Any, List, Optional


class TextDetector(BaseDetector):
    """文本数据质量检测器，集成配置管理和报告生成"""

    def __init__(self, cfg: Optional[Config] = None):
        super().__init__(cfg)
        self.texts = None

    def modality(self) -> str:
        return 'text'

    def detect(self, data: List[str], labels: Optional[List] = None) -> Dict[str, Any]:
        self.texts = data
        self.issues = []

        self._init_report({
            'name': self.cfg.get('data.dataset_name', 'unknown'),
            'total_samples': len(data),
        })

        empty_section = DiagnosisSection("empty_texts", "空文本检测")
        empty_texts = [i for i, text in enumerate(data) if not text or text.strip() == ""]
        for idx in empty_texts:
            self.issues.append({
                "type": "empty_text",
                "index": idx
            })
            empty_section.add_issue(IssueRecord(
                index=idx,
                issue_type="empty_text",
            ))

        empty_section.add_metric("empty_count", len(empty_texts))
        empty_section.add_metric("empty_rate", len(empty_texts) / len(data) * 100 if data else 0)
        self.report.add_section(empty_section)
        self._set_noise_rate("empty_text", len(empty_texts), len(data))

        duplicate_section = DiagnosisSection("duplicate_texts", "重复文本检测")
        seen = {}
        duplicate_texts = []
        for i, text in enumerate(data):
            text_clean = text.strip()
            if text_clean in seen:
                duplicate_texts.append({
                    "type": "duplicate_text",
                    "index": i,
                    "original_index": seen[text_clean]
                })
                duplicate_section.add_issue(IssueRecord(
                    index=i,
                    issue_type="duplicate_text",
                    details={"original_index": seen[text_clean]}
                ))
            else:
                seen[text_clean] = i

        self.issues.extend(duplicate_texts)
        duplicate_section.add_metric("duplicate_count", len(duplicate_texts))
        duplicate_section.add_metric("duplicate_rate", len(duplicate_texts) / len(data) * 100 if data else 0)
        self.report.add_section(duplicate_section)
        self._set_noise_rate("duplicate_text", len(duplicate_texts), len(data))

        length_section = DiagnosisSection("text_length", "文本长度分析")
        lengths = [len(text) for text in data]
        length_section.add_metric("average_length", float(np.mean(lengths)) if lengths else 0)
        length_section.add_metric("min_length", float(np.min(lengths)) if lengths else 0)
        length_section.add_metric("max_length", float(np.max(lengths)) if lengths else 0)
        length_section.add_metric("std_length", float(np.std(lengths)) if lengths else 0)

        if lengths:
            q1 = np.percentile(lengths, 25)
            q3 = np.percentile(lengths, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            abnormal_length_indices = [
                i for i, l in enumerate(lengths) if l < lower or l > upper
            ]
            for idx in abnormal_length_indices:
                self.issues.append({
                    "type": "abnormal_length",
                    "index": idx,
                    "length": lengths[idx]
                })
                length_section.add_issue(IssueRecord(
                    index=idx,
                    issue_type="abnormal_length",
                    details={"length": lengths[idx], "bounds": {"lower": float(lower), "upper": float(upper)}}
                ))
            self._set_noise_rate("abnormal_length", len(abnormal_length_indices), len(data))

        self.report.add_section(length_section)

        label_issues = {}
        if labels is not None and len(labels) == len(data):
            label_issues = self._detect_text_label_issues(data, labels)

        self.metrics = {
            "total_texts": len(data),
            "empty_text_rate": float(len(empty_texts) / len(data) * 100),
            "duplicate_text_rate": float(len(duplicate_texts) / len(data) * 100),
            "average_length": float(np.mean(lengths)) if lengths else 0,
            "min_length": float(np.min(lengths)) if lengths else 0,
            "max_length": float(np.max(lengths)) if lengths else 0
        }

        self.report.build_summary()

        return {
            "metrics": self.metrics,
            "issues": self.issues,
            "label_issues": label_issues,
            "diagnosis_report": self.report.to_dict()
        }

    def _detect_text_label_issues(self, texts: List[str], labels: List) -> Dict[str, Any]:
        try:
            from sklearn.preprocessing import LabelEncoder as LE

            max_features = self.cfg.get('feature_extraction.max_features_tfidf', 10000)
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            X = vectorizer.fit_transform(texts).toarray()

            le = LE()
            y = le.fit_transform(labels)

            cv_folds = self.cfg.get('detection.cross_validation_folds', 5)
            n_estimators = self.cfg.get('detection.n_estimators', 200)
            random_state = self.cfg.get('detection.random_state', 42)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )

            pred_probs = cross_val_predict(model, X, y, cv=cv_folds, method='predict_proba')

            from cleanlab.rank import get_label_quality_scores
            label_quality_scores = get_label_quality_scores(y, pred_probs)

            low_quality_threshold = self.cfg.get('detection.threshold', 0.6)
            error_indices = np.where(label_quality_scores < low_quality_threshold)[0].tolist()

            label_section = DiagnosisSection("text_label_errors", "文本标签错误检测")
            label_section.add_metric("error_count", len(error_indices))
            label_section.add_metric("error_rate", len(error_indices) / len(y) if len(y) > 0 else 0)

            model.fit(X, y)
            predictions = model.predict(X)
            suggested_labels = {}

            for idx in error_indices:
                original_label = int(y[idx])
                suggested = int(predictions[idx])
                score = float(label_quality_scores[idx])

                self.issues.append({
                    "type": "label_error",
                    "index": int(idx),
                    "original_label": original_label,
                    "quality_score": score
                })

                label_section.add_issue(IssueRecord(
                    index=int(idx),
                    issue_type="label_error",
                    original_label=original_label,
                    suggested_label=suggested,
                    quality_score=score,
                ))

                suggested_labels[str(idx)] = suggested

                if self.cfg.get('report.include_suggestion', True):
                    self._add_cured_sample({
                        'index': int(idx),
                        'original_data': texts[idx][:200],
                        'original_label': original_label,
                        'suggested_label': suggested,
                        'quality_score': score,
                    })

            self.report.add_section(label_section)
            self._set_noise_rate("text_label_error", len(error_indices), len(y))

            return {
                "error_count": len(error_indices),
                "error_rate": len(error_indices) / len(y) if len(y) > 0 else 0,
                "error_indices": error_indices,
                "suggested_labels": suggested_labels,
                "label_quality_scores": label_quality_scores.tolist(),
            }

        except Exception as e:
            return {"error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_issues(self) -> List[Dict[str, Any]]:
        return self.issues
