import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .base_detector import BaseDetector
from .cleanlab_wrapper import CleanlabWrapper
from .config import Config
from .report import DiagnosisSection, IssueRecord
from typing import Dict, Any, List, Optional


class TableDetector(BaseDetector):
    """表格数据质量检测器，集成配置管理和报告生成"""

    def __init__(self, cfg: Optional[Config] = None):
        super().__init__(cfg)
        self.data = None

    def modality(self) -> str:
        return 'tabular'

    def detect(self, data: pd.DataFrame, label_col: str = None) -> Dict[str, Any]:
        self.data = data
        self.issues = []

        if label_col is None:
            label_col = self.cfg.get('data.label_column', None)

        self._init_report({
            'name': self.cfg.get('data.dataset_name', 'unknown'),
            'total_samples': len(data),
            'total_features': len(data.columns),
            'feature_list': data.columns.tolist(),
        })

        basic_info = {
            "rows": len(data),
            "columns": len(data.columns),
            "columns_list": data.columns.tolist()
        }

        missing_section = DiagnosisSection("missing_values", "缺失值检测")
        missing_values = data.isnull().sum()
        missing_total = missing_values.sum()
        missing_percentage = (missing_total / (len(data) * len(data.columns))) * 100
        missing_stats = {
            "total": int(missing_total),
            "percentage": float(missing_percentage),
            "per_column": {}
        }

        for col in data.columns:
            col_missing = int(missing_values[col])
            col_missing_percentage = (col_missing / len(data)) * 100 if len(data) > 0 else 0
            missing_stats["per_column"][col] = {
                "count": col_missing,
                "percentage": float(col_missing_percentage)
            }

            if col_missing > 0:
                self.issues.append({
                    "type": "missing_value",
                    "column": col,
                    "count": col_missing,
                    "percentage": float(col_missing_percentage)
                })
                missing_section.add_issue(IssueRecord(
                    index=-1,
                    issue_type="missing_value",
                    details={"column": col, "count": col_missing, "percentage": float(col_missing_percentage)}
                ))

        missing_section.add_metric("total_missing", int(missing_total))
        missing_section.add_metric("missing_rate", float(missing_percentage))
        self.report.add_section(missing_section)
        self._set_noise_rate("missing_value", int(missing_total), len(data) * len(data.columns))

        outlier_section = DiagnosisSection("outliers", "异常值检测")
        outliers = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = int(((data[col] < lower_bound) | (data[col] > upper_bound)).sum())
                if outlier_count > 0:
                    outliers[col] = {
                        "count": outlier_count,
                        "percentage": float(outlier_count / len(data) * 100),
                        "bounds": {
                            "lower": float(lower_bound),
                            "upper": float(upper_bound)
                        }
                    }
                    self.issues.append({
                        "type": "outlier",
                        "column": col,
                        "count": outlier_count,
                        "percentage": float(outlier_count / len(data) * 100)
                    })
                    outlier_section.add_issue(IssueRecord(
                        index=-1,
                        issue_type="outlier",
                        details={"column": col, "count": outlier_count, "percentage": float(outlier_count / len(data) * 100)}
                    ))

        total_outliers = sum(info["count"] for info in outliers.values())
        outlier_section.add_metric("total_outliers", total_outliers)
        self.report.add_section(outlier_section)
        self._set_noise_rate("outlier", total_outliers, len(data))

        duplicate_section = DiagnosisSection("duplicates", "重复行检测")
        duplicate_rows = data.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(data)) * 100 if len(data) > 0 else 0
        duplicate_stats = {
            "count": int(duplicate_rows),
            "percentage": float(duplicate_percentage)
        }

        if duplicate_rows > 0:
            self.issues.append({
                "type": "duplicate_rows",
                "count": int(duplicate_rows),
                "percentage": float(duplicate_percentage)
            })
            duplicate_section.add_issue(IssueRecord(
                index=-1,
                issue_type="duplicate_rows",
                details={"count": int(duplicate_rows), "percentage": float(duplicate_percentage)}
            ))

        duplicate_section.add_metric("duplicate_count", int(duplicate_rows))
        self.report.add_section(duplicate_section)
        self._set_noise_rate("duplicate_rows", int(duplicate_rows), len(data))

        data_types = {}
        for col in data.columns:
            data_types[col] = str(data[col].dtype)

        label_issues = {}
        if label_col and label_col in data.columns:
            label_issues = self.detect_label_issues_with_confidence(data, label_col)

        self.metrics = {
            "rows": len(data),
            "columns": len(data.columns),
            "missing_value_rate": float(missing_percentage),
            "duplicate_row_rate": float(duplicate_percentage),
            "outlier_rate": float(total_outliers / len(data) * 100) if len(data) > 0 else 0
        }

        self.report.build_summary()

        return {
            "basic_info": basic_info,
            "missing_stats": missing_stats,
            "outlier_stats": outliers,
            "duplicate_stats": duplicate_stats,
            "data_types": data_types,
            "label_issues": label_issues,
            "metrics": self.metrics,
            "issues": self.issues,
            "diagnosis_report": self.report.to_dict()
        }

    def detect_label_issues_with_confidence(self, df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
        try:
            X = df.drop(columns=[label_col]).copy()
            y = df[label_col].copy()

            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = X[col].fillna('missing')
                X[col] = X[col].astype(str)
                X[col] = le.fit_transform(X[col])

            numeric_cols = X.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                X[col] = X[col].fillna(X[col].mean())

            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

            if not pd.api.types.is_numeric_dtype(y):
                y = pd.Categorical(y).codes

            y = y.values
            X = X.values

            cv_folds = self.cfg.get('detection.cross_validation_folds', 5)
            n_estimators = self.cfg.get('detection.n_estimators', 200)
            max_depth = self.cfg.get('detection.max_depth', 10)
            min_samples_split = self.cfg.get('detection.min_samples_split', 5)
            random_state = self.cfg.get('detection.random_state', 42)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )

            pred_probs = cross_val_predict(
                model, X, y, cv=cv_folds, method='predict_proba'
            )

            percentile_threshold = self.cfg.get('detection.percentile_threshold', 85)
            classes = np.unique(y)
            thresholds = {}
            for cls in classes:
                class_probs = pred_probs[y == cls, cls]
                if len(class_probs) > 0:
                    thresholds[cls] = np.percentile(class_probs, percentile_threshold)
                else:
                    thresholds[cls] = 0.5

            n_classes = len(classes)
            C_confident = np.zeros((n_classes, n_classes), dtype=int)

            for i, (prob, true_label) in enumerate(zip(pred_probs, y)):
                pred_label = np.argmax(prob)
                if prob[pred_label] >= thresholds.get(pred_label, 0.5):
                    C_confident[true_label, pred_label] += 1

            class_counts = np.bincount(y, minlength=n_classes)
            prior = class_counts / len(y)
            noise_matrix = C_confident / class_counts[:, None] if any(class_counts > 0) else np.zeros((n_classes, n_classes))

            from cleanlab.rank import get_label_quality_scores
            label_quality_scores = get_label_quality_scores(y, pred_probs)

            low_quality_threshold = self.cfg.get('detection.threshold', 0.6)
            error_indices = np.where(label_quality_scores < low_quality_threshold)[0].tolist()

            label_section = DiagnosisSection("label_errors", "标签错误检测（置信学习）")
            label_section.add_metric("error_count", len(error_indices))
            label_section.add_metric("error_rate", len(error_indices) / len(y) if len(y) > 0 else 0)
            label_section.add_statistic("confidence_thresholds", thresholds)
            label_section.add_statistic("confident_joint_matrix", C_confident.tolist())
            label_section.add_statistic("noise_matrix", noise_matrix.tolist())

            suggested_labels = {}
            model.fit(X, y)
            predictions = model.predict(X)

            for idx in error_indices:
                original_label = int(y[idx])
                suggested = int(predictions[idx])
                score = float(label_quality_scores[idx])

                self.issues.append({
                    "type": "label_error",
                    "index": int(idx),
                    "column": label_col,
                    "original_label": original_label,
                    "quality_score": score
                })

                label_section.add_issue(IssueRecord(
                    index=int(idx),
                    issue_type="label_error",
                    original_label=original_label,
                    suggested_label=suggested,
                    quality_score=score,
                    details={"column": label_col}
                ))

                suggested_labels[str(idx)] = suggested

                if self.cfg.get('report.include_suggestion', True):
                    row_data = df.iloc[idx].to_dict()
                    sample = {
                        'index': int(idx),
                        'original_data': {k: str(v)[:200] for k, v in row_data.items()},
                        'original_label': original_label,
                        'suggested_label': suggested,
                        'quality_score': score,
                    }
                    self._add_cured_sample(sample)

            self.report.add_section(label_section)
            self._set_noise_rate("label_error", len(error_indices), len(y))

            label_issues = {
                "error_count": len(error_indices),
                "error_rate": len(error_indices) / len(y) if len(y) > 0 else 0,
                "error_indices": error_indices,
                "suggested_labels": suggested_labels,
                "label_quality_scores": label_quality_scores.tolist(),
                "confidence_thresholds": thresholds,
                "confident_joint_matrix": C_confident.tolist(),
                "noise_matrix": noise_matrix.tolist(),
                "debug_info": {
                    "classes": classes.tolist(),
                    "class_counts": class_counts.tolist(),
                    "prior": prior.tolist(),
                    "mean_quality_score": float(np.mean(label_quality_scores)),
                    "min_quality_score": float(np.min(label_quality_scores)),
                    "max_quality_score": float(np.max(label_quality_scores))
                }
            }

            return label_issues

        except Exception as e:
            return {
                "error": str(e)
            }

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_issues(self) -> List[Dict[str, Any]]:
        return self.issues
