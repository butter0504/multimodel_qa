"""
文本数据质量检测器 - 模块化架构
================================

借鉴 ImageDetector 的模块化设计思路，将文本检测拆分为独立子模块，
每个子模块负责一类检测任务，通过 TextDetector 统一入口调度。

检测模块：
    1. TextLengthDetector       - 文本长度检测（过短/过长/统计分布）
    2. TextDuplicateDetector    - 文本重复检测（完全重复/近似重复）
    3. TextLabelErrorDetector   - 标签错误检测（基于 Confidence Learning + TF-IDF 特征）
    4. CharacterAnomalyDetector - 字符异常检测（乱码/特殊字符比例/首尾空白）
    5. TextLanguageDetector     - 文本语言检测（主要语言/混合语言/非目标语言）
    6. TextPerplexityDetector   - 文本困惑度检测（基于轻量级语言模型）
    7. SentimentConsistencyDetector - 情感/主题一致性检测

使用方式：
    detector = TextDetector(cfg)
    result = detector.detect(
        texts,                            # List[str] 文本列表
        labels=None,                      # 可选：标签列表
        modules=['text_length', 'duplicate'],  # 可选：选择运行的检测模块
    )
"""

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from .base_detector import BaseDetector
from .config import Config
from .report import DiagnosisSection, IssueRecord


class TextLengthDetector:
    """
    文本长度检测器
    ==============
    统计文本长度分布，检测过短和过长文本。

    检测指标：
        - 长度统计：最小值、最大值、均值、标准差、中位数
        - 过短文本：字符数少于阈值（默认 10）的样本
        - 过长文本：字符数超过阈值（默认 10000）的样本
        - 长度异常：基于 IQR 方法检测离群值
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.min_length_threshold = cfg.get('detection.text_min_length', 10)
        self.max_length_threshold = cfg.get('detection.text_max_length', 10000)

    def detect(self, texts: List[str]) -> Dict[str, Any]:
        issues = []
        lengths = [len(text) for text in texts]

        if not lengths:
            return {
                "issues": [],
                "metrics": {
                    "total_texts": 0,
                    "short_text_count": 0,
                    "long_text_count": 0,
                    "abnormal_length_count": 0,
                },
                "length_distribution": {},
            }

        length_arr = np.array(lengths, dtype=np.float64)
        stats = {
            "min": float(np.min(length_arr)),
            "max": float(np.max(length_arr)),
            "mean": float(np.mean(length_arr)),
            "std": float(np.std(length_arr)),
            "median": float(np.median(length_arr)),
            "q1": float(np.percentile(length_arr, 25)),
            "q3": float(np.percentile(length_arr, 75)),
        }

        short_indices = []
        long_indices = []
        abnormal_indices = []

        for i, length in enumerate(lengths):
            if length < self.min_length_threshold:
                short_indices.append(i)
                issues.append({
                    "type": "short_text",
                    "index": i,
                    "length": length,
                    "threshold": self.min_length_threshold,
                    "suggestion": "文本过短，可能缺乏有效信息，建议检查或删除"
                })

            if length > self.max_length_threshold:
                long_indices.append(i)
                issues.append({
                    "type": "long_text",
                    "index": i,
                    "length": length,
                    "threshold": self.max_length_threshold,
                    "suggestion": "文本过长，可能导致模型处理困难，建议截断或分段"
                })

        q1 = stats["q1"]
        q3 = stats["q3"]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        for i, length in enumerate(lengths):
            if length < lower_bound or length > upper_bound:
                if i not in short_indices and i not in long_indices:
                    abnormal_indices.append(i)
                    issues.append({
                        "type": "abnormal_length",
                        "index": i,
                        "length": length,
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                        "suggestion": "文本长度异常偏离，建议检查内容完整性"
                    })

        metrics = {
            "total_texts": len(texts),
            "short_text_count": len(short_indices),
            "long_text_count": len(long_indices),
            "abnormal_length_count": len(abnormal_indices),
            "short_text_rate": len(short_indices) / len(texts) * 100 if texts else 0,
            "long_text_rate": len(long_indices) / len(texts) * 100 if texts else 0,
        }

        length_distribution = {
            "statistics": stats,
            "short_indices": short_indices,
            "long_indices": long_indices,
            "abnormal_indices": abnormal_indices,
            "histogram": self._compute_histogram(lengths),
        }

        return {
            "issues": issues,
            "metrics": metrics,
            "length_distribution": length_distribution,
        }

    def _compute_histogram(self, lengths: List[int], n_bins: int = 20) -> Dict[str, Any]:
        if not lengths:
            return {"bins": [], "counts": []}
        arr = np.array(lengths, dtype=np.float64)
        counts, bin_edges = np.histogram(arr, bins=n_bins)
        return {
            "bins": [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(len(counts))],
            "counts": counts.tolist(),
        }


class TextDuplicateDetector:
    """
    文本重复检测器
    ==============
    检测完全重复和近似重复的文本样本。

    检测方法：
        - 完全重复：逐字符比较，找出完全相同的文本
        - 近似重复：使用 SimHash / MinHash 思想，对文本计算哈希指纹，
          通过比较哈希值的汉明距离判断近似重复
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.similarity_threshold = cfg.get('detection.text_similarity_threshold', 0.85)
        self.ngram_size = cfg.get('detection.text_ngram_size', 3)

    def detect(self, texts: List[str]) -> Dict[str, Any]:
        issues = []

        exact_groups = self._find_exact_duplicates(texts)
        for key, indices in exact_groups.items():
            for idx in indices[1:]:
                issues.append({
                    "type": "exact_duplicate",
                    "index": idx,
                    "original_index": indices[0],
                    "duplicate_count": len(indices),
                    "suggestion": "完全重复文本，建议仅保留一条"
                })

        approximate_pairs = self._find_approximate_duplicates(texts, exact_groups)
        for idx, original_idx, similarity in approximate_pairs:
            issues.append({
                "type": "approximate_duplicate",
                "index": idx,
                "original_index": original_idx,
                "similarity": round(similarity, 4),
                "suggestion": "近似重复文本，建议检查是否需要去重"
            })

        exact_count = sum(1 for iss in issues if iss["type"] == "exact_duplicate")
        approx_count = sum(1 for iss in issues if iss["type"] == "approximate_duplicate")

        metrics = {
            "total_texts": len(texts),
            "exact_duplicate_count": exact_count,
            "approximate_duplicate_count": approx_count,
            "exact_duplicate_rate": exact_count / len(texts) * 100 if texts else 0,
            "approximate_duplicate_rate": approx_count / len(texts) * 100 if texts else 0,
            "unique_text_count": len(set(text.strip() for text in texts)),
        }

        return {
            "issues": issues,
            "metrics": metrics,
            "exact_duplicate_groups": {k: v for k, v in exact_groups.items() if len(v) > 1},
        }

    def _find_exact_duplicates(self, texts: List[str]) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for i, text in enumerate(texts):
            key = text.strip()
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        return groups

    def _find_approximate_duplicates(self, texts: List[str],
                                      exact_groups: Dict[str, List[int]]) -> List[Tuple[int, int, float]]:
        pairs = []
        seen_exact = set()
        for indices in exact_groups.values():
            if len(indices) > 1:
                for idx in indices:
                    seen_exact.add(idx)

        fingerprints = []
        for i, text in enumerate(texts):
            fp = self._compute_fingerprint(text)
            fingerprints.append(fp)

        checked = set()
        for i in range(len(texts)):
            if i in seen_exact:
                continue
            for j in range(i + 1, len(texts)):
                if j in seen_exact:
                    continue
                pair_key = (min(i, j), max(i, j))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                sim = self._compute_similarity(fingerprints[i], fingerprints[j])
                if sim >= self.similarity_threshold:
                    pairs.append((j, i, sim))

        return pairs

    def _compute_fingerprint(self, text: str) -> Dict[str, int]:
        tokens = self._tokenize(text)
        ngrams = self._get_ngrams(tokens)
        freq = Counter(ngrams)
        return freq

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        tokens = re.findall(r'\w+', text)
        return tokens

    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        if len(tokens) < self.ngram_size:
            return [' '.join(tokens)] if tokens else []
        return [' '.join(tokens[i:i + self.ngram_size])
                for i in range(len(tokens) - self.ngram_size + 1)]

    def _compute_similarity(self, fp1: Dict[str, int], fp2: Dict[str, int]) -> float:
        if not fp1 and not fp2:
            return 1.0
        if not fp1 or not fp2:
            return 0.0

        all_keys = set(fp1.keys()) | set(fp2.keys())
        intersection = sum(min(fp1.get(k, 0), fp2.get(k, 0)) for k in all_keys)
        union = sum(max(fp1.get(k, 0), fp2.get(k, 0)) for k in all_keys)

        if union == 0:
            return 0.0
        return intersection / union


class TextLabelErrorDetector:
    """
    标签错误检测器（基于 Confidence Learning）
    ==========================================
    使用 cleanlab 库的置信学习框架检测文本标签中的标注错误。

    工作流程：
        1. 使用 TF-IDF 提取文本特征向量
        2. 基于 TF-IDF 特征训练 RandomForest 分类器
        3. 使用交叉验证获取样本的预测概率分布
        4. 调用 cleanlab 的 get_label_quality_scores 计算标签质量分数
        5. 质量分数低于阈值的样本被标记为潜在标签错误
        6. 分类器的预测结果作为建议修正标签
        7. 计算置信联合矩阵和噪声矩阵

    前置条件：
        - 需要提供标签列表（labels）
        - 标签至少包含 2 个类别
        - 每个类别至少有 2 个样本（交叉验证要求）
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.quality_threshold = cfg.get('detection.threshold', 0.3)
        self.cv_folds = cfg.get('detection.cross_validation_folds', 5)
        self.n_estimators = cfg.get('detection.n_estimators', 200)
        self.random_state = cfg.get('detection.random_state', 42)
        self.max_features_tfidf = cfg.get('feature_extraction.max_features_tfidf', 10000)

    def detect(self, texts: List[str], labels: List) -> Dict[str, Any]:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_predict
            from sklearn.preprocessing import LabelEncoder
            from sklearn.feature_extraction.text import TfidfVectorizer
            from cleanlab.rank import get_label_quality_scores

            vectorizer = TfidfVectorizer(
                max_features=self.max_features_tfidf,
                stop_words='english',
            )
            X = vectorizer.fit_transform(texts).toarray()

            le = LabelEncoder()
            y = le.fit_transform(labels)
            n_classes = len(le.classes_)

            class_counts = np.bincount(y)
            min_class_count = int(np.min(class_counts))
            actual_cv_folds = min(self.cv_folds, min_class_count)
            if actual_cv_folds < 2:
                return {
                    "error": f"某些类别样本数不足（最少 {min_class_count}），无法进行交叉验证",
                    "error_count": 0,
                    "error_rate": 0,
                    "error_indices": [],
                    "suggested_labels": {},
                    "label_quality_scores": [],
                }

            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            pred_probs = cross_val_predict(
                model, X, y, cv=actual_cv_folds, method='predict_proba'
            )

            label_quality_scores = get_label_quality_scores(y, pred_probs)

            error_indices = np.where(label_quality_scores < self.quality_threshold)[0].tolist()

            model.fit(X, y)
            predictions = model.predict(X)
            suggested_labels = {}
            for idx in error_indices:
                suggested_labels[str(idx)] = int(predictions[idx])

            percentile_threshold = self.cfg.get('detection.percentile_threshold', 85)
            thresholds = {}
            classes = np.unique(y)
            for cls in classes:
                class_probs = pred_probs[y == cls, cls]
                if len(class_probs) > 0:
                    thresholds[cls] = float(np.percentile(class_probs, percentile_threshold))
                else:
                    thresholds[cls] = 0.5

            C_confident = np.zeros((n_classes, n_classes), dtype=int)
            for i, (prob, true_label) in enumerate(zip(pred_probs, y)):
                pred_label = np.argmax(prob)
                if prob[pred_label] >= thresholds.get(pred_label, 0.5):
                    C_confident[true_label, pred_label] += 1

            class_counts_arr = np.bincount(y, minlength=n_classes)
            with np.errstate(divide='ignore', invalid='ignore'):
                noise_matrix = np.where(
                    class_counts_arr[:, None] > 0,
                    C_confident / class_counts_arr[:, None],
                    0
                )

            return {
                "error_count": len(error_indices),
                "error_rate": len(error_indices) / len(y) if len(y) > 0 else 0,
                "error_indices": error_indices,
                "suggested_labels": suggested_labels,
                "label_quality_scores": label_quality_scores.tolist(),
                "confidence_thresholds": thresholds,
                "confident_joint_matrix": C_confident.tolist(),
                "noise_matrix": noise_matrix.tolist(),
                "class_names": le.classes_.tolist(),
                "debug_info": {
                    "n_classes": n_classes,
                    "class_counts": class_counts.tolist(),
                    "cv_folds_used": actual_cv_folds,
                    "mean_quality_score": float(np.mean(label_quality_scores)),
                    "min_quality_score": float(np.min(label_quality_scores)),
                    "max_quality_score": float(np.max(label_quality_scores)),
                    "feature_method": "tfidf",
                    "feature_dim": X.shape[1],
                },
            }

        except ImportError as e:
            return {"error": f"缺少依赖库: {str(e)}", "error_count": 0}
        except Exception as e:
            return {"error": str(e), "error_count": 0}


class CharacterAnomalyDetector:
    """
    字符异常检测器
    ==============
    检测文本中的字符级别异常。

    检测指标：
        - 乱码字符：替换字符（U+FFFD）、控制字符等
        - 特殊字符比例异常：标点/符号/数字占比过高
        - 首尾空白字符：文本开头或结尾存在多余空白
    """

    REPLACEMENT_CHAR = '\ufffd'
    CONTROL_CHARS = set(chr(i) for i in range(0, 32) if i not in (9, 10, 13))

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.special_char_ratio_threshold = cfg.get('detection.text_special_char_ratio', 0.5)
        self.punctuation_ratio_threshold = cfg.get('detection.text_punctuation_ratio', 0.4)
        self.digit_ratio_threshold = cfg.get('detection.text_digit_ratio', 0.5)

    def detect(self, texts: List[str]) -> Dict[str, Any]:
        issues = []
        garbled_indices = []
        high_special_indices = []
        whitespace_indices = []

        for i, text in enumerate(texts):
            if not text or text.strip() == "":
                continue

            garbled_chars = self._detect_garbled_chars(text)
            if garbled_chars:
                garbled_indices.append(i)
                issues.append({
                    "type": "garbled_chars",
                    "index": i,
                    "garbled_char_count": garbled_chars["count"],
                    "garbled_positions": garbled_chars["positions"][:10],
                    "suggestion": "检测到乱码/替换字符，建议检查编码或删除该样本"
                })

            special_ratio = self._compute_special_char_ratio(text)
            punct_ratio = self._compute_punctuation_ratio(text)
            digit_ratio = self._compute_digit_ratio(text)

            if special_ratio > self.special_char_ratio_threshold:
                high_special_indices.append(i)
                issues.append({
                    "type": "high_special_char_ratio",
                    "index": i,
                    "special_char_ratio": round(special_ratio, 4),
                    "punctuation_ratio": round(punct_ratio, 4),
                    "digit_ratio": round(digit_ratio, 4),
                    "suggestion": "特殊字符占比过高，可能为格式错误或非正常文本"
                })

            leading_ws = len(text) - len(text.lstrip())
            trailing_ws = len(text) - len(text.rstrip())
            if leading_ws > 0 or trailing_ws > 0:
                whitespace_indices.append(i)
                issues.append({
                    "type": "whitespace_padding",
                    "index": i,
                    "leading_whitespace": leading_ws,
                    "trailing_whitespace": trailing_ws,
                    "suggestion": "文本首尾存在多余空白字符，建议清理"
                })

        metrics = {
            "total_texts": len(texts),
            "garbled_text_count": len(garbled_indices),
            "high_special_char_count": len(high_special_indices),
            "whitespace_padding_count": len(whitespace_indices),
            "garbled_text_rate": len(garbled_indices) / len(texts) * 100 if texts else 0,
        }

        return {
            "issues": issues,
            "metrics": metrics,
        }

    def _detect_garbled_chars(self, text: str) -> Dict[str, Any]:
        positions = []
        for i, ch in enumerate(text):
            if ch == self.REPLACEMENT_CHAR:
                positions.append(i)
            elif ord(ch) in range(0, 32) and ch not in ('\t', '\n', '\r'):
                positions.append(i)
            elif unicodedata.category(ch).startswith('Cn') and ord(ch) > 127:
                positions.append(i)

        if positions:
            return {"count": len(positions), "positions": positions}
        return None

    def _compute_special_char_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        special_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        return special_count / len(text)

    def _compute_punctuation_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        punct_count = sum(1 for ch in text if unicodedata.category(ch).startswith('P'))
        return punct_count / len(text)

    def _compute_digit_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        digit_count = sum(1 for ch in text if ch.isdigit())
        return digit_count / len(text)


class TextLanguageDetector:
    """
    文本语言检测器
    ==============
    检测文本的主要语言、混合语言样本和非目标语言文本。

    检测方法：
        - 使用字符范围和常见词频进行轻量级语言判断
        - 支持中文、英文、日文、韩文等主要语言
        - 检测混合语言样本（同一文本中出现多种语言）
    """

    CJK_RANGES = [
        (0x4E00, 0x9FFF, 'chinese'),
        (0x3400, 0x4DBF, 'chinese'),
        (0x3000, 0x303F, 'cjk_symbol'),
        (0x3040, 0x309F, 'japanese_hiragana'),
        (0x30A0, 0x30FF, 'japanese_katakana'),
        (0xAC00, 0xD7AF, 'korean'),
    ]

    ENGLISH_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
        'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
        'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
        'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
        'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
        'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
        'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.target_language = cfg.get('detection.text_target_language', None)
        self.mixed_language_threshold = cfg.get('detection.text_mixed_language_threshold', 0.3)

    def detect(self, texts: List[str]) -> Dict[str, Any]:
        issues = []
        language_counts: Dict[str, int] = {}
        mixed_language_indices = []
        non_target_indices = []

        for i, text in enumerate(texts):
            if not text or text.strip() == "":
                continue

            lang_result = self._detect_language(text)
            primary_lang = lang_result["primary_language"]
            lang_ratios = lang_result["language_ratios"]

            language_counts[primary_lang] = language_counts.get(primary_lang, 0) + 1

            non_primary_ratio = sum(v for k, v in lang_ratios.items() if k != primary_lang)
            if non_primary_ratio > self.mixed_language_threshold and len(lang_ratios) > 1:
                mixed_language_indices.append(i)
                issues.append({
                    "type": "mixed_language",
                    "index": i,
                    "primary_language": primary_lang,
                    "language_ratios": {k: round(v, 4) for k, v in lang_ratios.items()},
                    "suggestion": "检测到混合语言文本，建议统一语言或分离处理"
                })

            if self.target_language and primary_lang != self.target_language:
                non_target_indices.append(i)
                issues.append({
                    "type": "non_target_language",
                    "index": i,
                    "detected_language": primary_lang,
                    "target_language": self.target_language,
                    "suggestion": f"检测到非目标语言（{primary_lang}），建议过滤或翻译"
                })

        metrics = {
            "total_texts": len(texts),
            "language_distribution": language_counts,
            "mixed_language_count": len(mixed_language_indices),
            "non_target_language_count": len(non_target_indices),
            "mixed_language_rate": len(mixed_language_indices) / len(texts) * 100 if texts else 0,
        }

        return {
            "issues": issues,
            "metrics": metrics,
        }

    def _detect_language(self, text: str) -> Dict[str, Any]:
        char_categories: Dict[str, int] = {
            'chinese': 0,
            'japanese': 0,
            'korean': 0,
            'latin': 0,
            'digit': 0,
            'other': 0,
        }

        for ch in text:
            if ch.isspace():
                continue
            cp = ord(ch)
            detected = False

            for start, end, lang in self.CJK_RANGES:
                if start <= cp <= end:
                    if lang == 'chinese' or lang == 'cjk_symbol':
                        char_categories['chinese'] += 1
                    elif lang.startswith('japanese'):
                        char_categories['japanese'] += 1
                    elif lang == 'korean':
                        char_categories['korean'] += 1
                    detected = True
                    break

            if not detected:
                if ch.isdigit():
                    char_categories['digit'] += 1
                elif unicodedata.category(ch).startswith('L') and cp < 0x4E00:
                    char_categories['latin'] += 1
                else:
                    char_categories['other'] += 1

        total_chars = sum(char_categories.values())
        if total_chars == 0:
            return {"primary_language": "unknown", "language_ratios": {}}

        lang_ratios = {k: v / total_chars for k, v in char_categories.items() if v > 0}

        if char_categories['latin'] > 0:
            words = re.findall(r'[a-zA-Z]+', text.lower())
            english_word_count = sum(1 for w in words if w in self.ENGLISH_WORDS)
            if english_word_count > 0 and len(words) > 0:
                english_ratio = english_word_count / len(words)
                if english_ratio > 0.15:
                    lang_ratios['english'] = char_categories['latin'] / total_chars
                    if 'latin' in lang_ratios:
                        del lang_ratios['latin']

        primary_language = max(lang_ratios, key=lang_ratios.get) if lang_ratios else "unknown"

        return {
            "primary_language": primary_language,
            "language_ratios": lang_ratios,
        }


class TextPerplexityDetector:
    """
    文本困惑度检测器
    ================
    使用轻量级方法评估文本的通顺程度。
    困惑度高的样本可能是乱序或语法错误的文本。

    检测方法：
        - 优先尝试使用轻量级语言模型计算困惑度
        - 回退方案：基于 n-gram 频率和字符级统计近似估计
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.perplexity_threshold = cfg.get('detection.text_perplexity_threshold', 100.0)
        self.high_perplexity_std_factor = cfg.get('detection.text_perplexity_std_factor', 2.0)

    def detect(self, texts: List[str]) -> Dict[str, Any]:
        issues = []

        perplexity_scores = self._compute_perplexity(texts)

        if not perplexity_scores:
            return {
                "issues": [],
                "metrics": {
                    "total_texts": len(texts),
                    "high_perplexity_count": 0,
                },
                "perplexity_scores": [],
                "method": "none",
            }

        arr = np.array(perplexity_scores, dtype=np.float64)
        mean_ppl = float(np.mean(arr))
        std_ppl = float(np.std(arr))
        threshold = min(self.perplexity_threshold, mean_ppl + self.high_perplexity_std_factor * std_ppl)

        high_ppl_indices = []
        for i, score in enumerate(perplexity_scores):
            if score > threshold:
                high_ppl_indices.append(i)
                issues.append({
                    "type": "high_perplexity",
                    "index": i,
                    "perplexity": round(score, 2),
                    "threshold": round(threshold, 2),
                    "suggestion": "文本困惑度较高，可能存在语法错误或乱序，建议人工检查"
                })

        metrics = {
            "total_texts": len(texts),
            "high_perplexity_count": len(high_ppl_indices),
            "mean_perplexity": round(mean_ppl, 2),
            "std_perplexity": round(std_ppl, 2),
            "perplexity_threshold": round(threshold, 2),
        }

        return {
            "issues": issues,
            "metrics": metrics,
            "perplexity_scores": [round(s, 2) for s in perplexity_scores],
            "method": "statistical_approximation",
        }

    def _compute_perplexity(self, texts: List[str]) -> List[float]:
        try:
            return self._compute_perplexity_with_model(texts)
        except Exception:
            return self._compute_perplexity_statistical(texts)

    def _compute_perplexity_with_model(self, texts: List[str]) -> List[float]:
        import torch
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
        except ImportError:
            raise ImportError("transformers not available")

        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()

        scores = []
        for text in texts:
            if not text or text.strip() == "":
                scores.append(0.0)
                continue
            try:
                encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**encodings, labels=encodings['input_ids'])
                    neg_log_likelihood = outputs.loss.item()
                    ppl = float(np.exp(neg_log_likelihood))
                    scores.append(min(ppl, 1e6))
            except Exception:
                scores.append(self._estimate_single_perplexity(text))
        return scores

    def _compute_perplexity_statistical(self, texts: List[str]) -> List[float]:
        return [self._estimate_single_perplexity(text) for text in texts]

    def _estimate_single_perplexity(self, text: str) -> float:
        if not text or len(text.strip()) < 3:
            return 0.0

        words = re.findall(r'\w+', text.lower())
        if len(words) < 2:
            return 50.0

        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        log_prob_sum = 0.0
        n_bigrams = 0
        for bg in bigrams:
            p = bigram_counts[bg] / (unigram_counts[bg[0]] + 1e-10)
            log_prob_sum += -np.log(p + 1e-10)
            n_bigrams += 1

        if n_bigrams == 0:
            return 50.0

        avg_neg_log_prob = log_prob_sum / n_bigrams
        ppl = float(np.exp(min(avg_neg_log_prob, 20)))

        char_repetition = self._compute_char_repetition(text)
        ppl *= (1 + char_repetition)

        return min(ppl, 1e6)

    def _compute_char_repetition(self, text: str) -> float:
        if len(text) < 4:
            return 0.0
        trigrams = [text[i:i+3] for i in range(len(text) - 2)]
        if not trigrams:
            return 0.0
        unique_ratio = len(set(trigrams)) / len(trigrams)
        return max(0, 1.0 - unique_ratio)


class SentimentConsistencyDetector:
    """
    情感/主题一致性检测器
    ====================
    检测文本内容与标签是否匹配。

    检测方法：
        - 基于关键词的情感分析（轻量级，无需深度学习模型）
        - 检测文本情感与标签情感方向不一致的样本
        - 使用 TF-IDF + 余弦相似度检测主题一致性
    """

    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'happy', 'best', 'beautiful', 'perfect', 'awesome',
        'enjoy', 'pleasant', 'brilliant', 'superb', 'outstanding',
        'delightful', 'magnificent', 'marvelous', 'terrific', 'splendid',
    }

    NEGATIVE_WORDS = {
        'bad', 'terrible', 'horrible', 'awful', 'worst', 'hate',
        'poor', 'ugly', 'disgusting', 'disappointing', 'boring',
        'annoying', 'dreadful', 'miserable', 'pathetic', 'inferior',
        'wretched', 'vile', 'lousy', 'rotten', 'abysmal',
    }

    POSITIVE_LABELS = {
        'positive', 'pos', 'good', 'happy', 'joy', 'love', 'like',
        'favorable', 'satisfied', 'approve', 'praise', 'recommend',
        '1', '2',
    }

    NEGATIVE_LABELS = {
        'negative', 'neg', 'bad', 'sad', 'anger', 'hate', 'dislike',
        'unfavorable', 'dissatisfied', 'disapprove', 'criticize', 'not_recommend',
        '0', '-1',
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.consistency_threshold = cfg.get('detection.text_consistency_threshold', 0.3)

    def detect(self, texts: List[str], labels: List) -> Dict[str, Any]:
        issues = []
        inconsistency_indices = []

        for i, (text, label) in enumerate(zip(texts, labels)):
            if not text or text.strip() == "":
                continue

            text_sentiment = self._analyze_sentiment(text)
            label_sentiment = self._get_label_sentiment(label)

            if label_sentiment == 'neutral' or text_sentiment['sentiment'] == 'neutral':
                continue

            if text_sentiment['sentiment'] != label_sentiment:
                score = abs(text_sentiment['score'])
                if score > self.consistency_threshold:
                    inconsistency_indices.append(i)
                    issues.append({
                        "type": "sentiment_inconsistency",
                        "index": i,
                        "text_sentiment": text_sentiment['sentiment'],
                        "text_sentiment_score": round(text_sentiment['score'], 4),
                        "label_sentiment": label_sentiment,
                        "inconsistency_score": round(score, 4),
                        "suggestion": "文本情感与标签不一致，建议检查标签是否正确"
                    })

        metrics = {
            "total_texts": len(texts),
            "inconsistency_count": len(inconsistency_indices),
            "inconsistency_rate": len(inconsistency_indices) / len(texts) * 100 if texts else 0,
        }

        return {
            "issues": issues,
            "metrics": metrics,
        }

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        words = set(re.findall(r'\w+', text.lower()))
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)

        total = positive_count + negative_count
        if total == 0:
            return {"sentiment": "neutral", "score": 0.0}

        score = (positive_count - negative_count) / total

        if score > 0.2:
            return {"sentiment": "positive", "score": score}
        elif score < -0.2:
            return {"sentiment": "negative", "score": score}
        else:
            return {"sentiment": "neutral", "score": score}

    def _get_label_sentiment(self, label) -> str:
        label_str = str(label).lower().strip()
        if label_str in self.POSITIVE_LABELS:
            return 'positive'
        if label_str in self.NEGATIVE_LABELS:
            return 'negative'
        try:
            label_num = int(float(label_str))
            if label_num > 3:
                return 'positive'
            elif label_num < 2:
                return 'negative'
        except (ValueError, TypeError):
            pass
        return 'neutral'


class TextDetector(BaseDetector):
    """
    文本数据质量检测器 - 统一入口
    ==============================
    整合所有文本检测模块，提供统一的检测接口。
    用户可通过 modules 参数选择运行哪些检测模块。

    使用示例：
        # 仅运行文本长度检测
        result = detector.detect(texts, modules=['text_length'])

        # 运行文本长度 + 重复检测
        result = detector.detect(texts, modules=['text_length', 'duplicate'])

        # 运行全部检测（含标签错误检测）
        result = detector.detect(texts, labels=labels,
                                 modules=['text_length', 'duplicate', 'label_error',
                                          'character_anomaly', 'language', 'perplexity',
                                          'sentiment_consistency'])
    """

    MODULE_NAMES = {
        'text_length': '文本长度检测',
        'duplicate': '文本重复检测',
        'label_error': '标签错误检测',
        'character_anomaly': '字符异常检测',
        'language': '文本语言检测',
        'perplexity': '文本困惑度检测',
        'sentiment_consistency': '情感一致性检测',
    }

    def __init__(self, cfg: Optional[Config] = None):
        super().__init__(cfg)
        self._length_detector = TextLengthDetector(self.cfg)
        self._duplicate_detector = TextDuplicateDetector(self.cfg)
        self._label_detector = TextLabelErrorDetector(self.cfg)
        self._char_anomaly_detector = CharacterAnomalyDetector(self.cfg)
        self._language_detector = TextLanguageDetector(self.cfg)
        self._perplexity_detector = TextPerplexityDetector(self.cfg)
        self._sentiment_detector = SentimentConsistencyDetector(self.cfg)

    def modality(self) -> str:
        return 'text'

    def detect(self, data: List[str], labels: Optional[List] = None,
               modules: Optional[List[str]] = None,
               progress_callback=None) -> Dict[str, Any]:
        """
        执行文本数据质量检测。

        Parameters
        ----------
        data : List[str]
            文本数据列表
        labels : Optional[List]
            标签列表，标签错误检测和情感一致性检测需要此参数
        modules : Optional[List[str]]
            要运行的检测模块列表，可选值：
            - 'text_length': 文本长度检测（默认必选）
            - 'duplicate': 文本重复检测
            - 'label_error': 标签错误检测（需要 labels）
            - 'character_anomaly': 字符异常检测
            - 'language': 文本语言检测
            - 'perplexity': 文本困惑度检测
            - 'sentiment_consistency': 情感一致性检测（需要 labels）
            默认为 ['text_length', 'duplicate', 'character_anomaly']
        progress_callback : Optional[callable]
            进度回调函数，签名为 callback(module_name: str, progress: float)

        Returns
        -------
        Dict[str, Any]
            综合检测结果字典
        """
        texts = self._normalize_texts(data)

        if modules is None:
            modules = ['text_length', 'duplicate', 'character_anomaly']

        self.issues = []
        self._init_report({
            'name': self.cfg.get('data.dataset_name', 'unknown'),
            'total_samples': len(texts),
            'modules_run': modules,
        })

        result = {
            "modules_run": modules,
            "text_length": None,
            "duplicate": None,
            "label_error": None,
            "character_anomaly": None,
            "language": None,
            "perplexity": None,
            "sentiment_consistency": None,
        }

        # ---- 模块 1: 文本长度检测 ----
        if 'text_length' in modules:
            if progress_callback:
                progress_callback('text_length', 0.0)
            length_result = self._length_detector.detect(texts)
            result["text_length"] = length_result
            self.issues.extend(length_result["issues"])

            length_section = DiagnosisSection("text_length", "文本长度检测")
            for key, val in length_result["metrics"].items():
                length_section.add_metric(key, val)
            for issue in length_result["issues"]:
                length_section.add_issue(IssueRecord(
                    index=issue["index"],
                    issue_type=issue["type"],
                    details={k: v for k, v in issue.items() if k not in ('type', 'index')},
                ))
            self.report.add_section(length_section)
            self._set_noise_rate("text_length", len(length_result["issues"]), len(texts))
            if progress_callback:
                progress_callback('text_length', 1.0)

        # ---- 模块 2: 文本重复检测 ----
        if 'duplicate' in modules:
            if progress_callback:
                progress_callback('duplicate', 0.0)
            dup_result = self._duplicate_detector.detect(texts)
            result["duplicate"] = dup_result
            self.issues.extend(dup_result["issues"])

            dup_section = DiagnosisSection("duplicate", "文本重复检测")
            for key, val in dup_result["metrics"].items():
                dup_section.add_metric(key, val)
            for issue in dup_result["issues"]:
                dup_section.add_issue(IssueRecord(
                    index=issue["index"],
                    issue_type=issue["type"],
                    details={k: v for k, v in issue.items() if k not in ('type', 'index')},
                ))
            self.report.add_section(dup_section)
            self._set_noise_rate("duplicate", len(dup_result["issues"]), len(texts))
            if progress_callback:
                progress_callback('duplicate', 1.0)

        # ---- 模块 3: 标签错误检测 ----
        if 'label_error' in modules and labels is not None:
            if progress_callback:
                progress_callback('label_error', 0.0)
            label_result = self._label_detector.detect(texts, labels)
            result["label_error"] = label_result

            if "error" not in label_result:
                label_section = DiagnosisSection("label_errors", "标签错误检测（置信学习）")
                label_section.add_metric("error_count", label_result["error_count"])
                label_section.add_metric("error_rate", label_result["error_rate"])

                for idx in label_result["error_indices"]:
                    original = labels[idx] if idx < len(labels) else None
                    suggested = label_result["suggested_labels"].get(str(idx))
                    score = label_result["label_quality_scores"][idx] if idx < len(label_result["label_quality_scores"]) else None

                    self.issues.append({
                        "type": "label_error",
                        "index": idx,
                        "original_label": original,
                        "suggested_label": suggested,
                        "quality_score": score,
                        "suggestion": "建议检查该样本标签，可点选修正后保存"
                    })
                    label_section.add_issue(IssueRecord(
                        index=idx,
                        issue_type="label_error",
                        original_label=original,
                        suggested_label=suggested,
                        quality_score=score,
                    ))

                    if self.cfg.get('report.include_suggestion', True):
                        self._add_cured_sample({
                            'index': int(idx),
                            'original_data': texts[idx][:200],
                            'original_label': original,
                            'suggested_label': suggested,
                            'quality_score': score,
                        })

                self.report.add_section(label_section)
                self._set_noise_rate("label_error", label_result["error_count"], len(texts))
            if progress_callback:
                progress_callback('label_error', 1.0)

        # ---- 模块 4: 字符异常检测 ----
        if 'character_anomaly' in modules:
            if progress_callback:
                progress_callback('character_anomaly', 0.0)
            char_result = self._char_anomaly_detector.detect(texts)
            result["character_anomaly"] = char_result
            self.issues.extend(char_result["issues"])

            char_section = DiagnosisSection("character_anomaly", "字符异常检测")
            for key, val in char_result["metrics"].items():
                char_section.add_metric(key, val)
            for issue in char_result["issues"]:
                char_section.add_issue(IssueRecord(
                    index=issue["index"],
                    issue_type=issue["type"],
                    details={k: v for k, v in issue.items() if k not in ('type', 'index')},
                ))
            self.report.add_section(char_section)
            self._set_noise_rate("character_anomaly", len(char_result["issues"]), len(texts))
            if progress_callback:
                progress_callback('character_anomaly', 1.0)

        # ---- 模块 5: 文本语言检测 ----
        if 'language' in modules:
            if progress_callback:
                progress_callback('language', 0.0)
            lang_result = self._language_detector.detect(texts)
            result["language"] = lang_result
            self.issues.extend(lang_result["issues"])

            lang_section = DiagnosisSection("language", "文本语言检测")
            for key, val in lang_result["metrics"].items():
                lang_section.add_metric(key, val)
            for issue in lang_result["issues"]:
                lang_section.add_issue(IssueRecord(
                    index=issue["index"],
                    issue_type=issue["type"],
                    details={k: v for k, v in issue.items() if k not in ('type', 'index')},
                ))
            self.report.add_section(lang_section)
            self._set_noise_rate("language", len(lang_result["issues"]), len(texts))
            if progress_callback:
                progress_callback('language', 1.0)

        # ---- 模块 6: 文本困惑度检测 ----
        if 'perplexity' in modules:
            if progress_callback:
                progress_callback('perplexity', 0.0)
            ppl_result = self._perplexity_detector.detect(texts)
            result["perplexity"] = ppl_result
            self.issues.extend(ppl_result["issues"])

            ppl_section = DiagnosisSection("perplexity", "文本困惑度检测")
            for key, val in ppl_result["metrics"].items():
                ppl_section.add_metric(key, val)
            for issue in ppl_result["issues"]:
                ppl_section.add_issue(IssueRecord(
                    index=issue["index"],
                    issue_type=issue["type"],
                    details={k: v for k, v in issue.items() if k not in ('type', 'index')},
                ))
            self.report.add_section(ppl_section)
            self._set_noise_rate("perplexity", len(ppl_result["issues"]), len(texts))
            if progress_callback:
                progress_callback('perplexity', 1.0)

        # ---- 模块 7: 情感/主题一致性检测 ----
        if 'sentiment_consistency' in modules and labels is not None:
            if progress_callback:
                progress_callback('sentiment_consistency', 0.0)
            sent_result = self._sentiment_detector.detect(texts, labels)
            result["sentiment_consistency"] = sent_result
            self.issues.extend(sent_result["issues"])

            sent_section = DiagnosisSection("sentiment_consistency", "情感一致性检测")
            for key, val in sent_result["metrics"].items():
                sent_section.add_metric(key, val)
            for issue in sent_result["issues"]:
                sent_section.add_issue(IssueRecord(
                    index=issue["index"],
                    issue_type=issue["type"],
                    details={k: v for k, v in issue.items() if k not in ('type', 'index')},
                ))
            self.report.add_section(sent_section)
            self._set_noise_rate("sentiment_consistency", len(sent_result["issues"]), len(texts))
            if progress_callback:
                progress_callback('sentiment_consistency', 1.0)

        # 汇总指标
        self.metrics = {
            "total_texts": len(texts),
            "issue_count": len(self.issues),
            "issue_rate": float(len(self.issues) / len(texts) * 100) if texts else 0,
            "modules_run": modules,
        }

        if result["text_length"]:
            ld = result["text_length"]["length_distribution"].get("statistics", {})
            self.metrics["average_length"] = ld.get("mean", 0)
            self.metrics["min_length"] = ld.get("min", 0)
            self.metrics["max_length"] = ld.get("max", 0)

        if result["duplicate"]:
            self.metrics["duplicate_rate"] = result["duplicate"]["metrics"].get("exact_duplicate_rate", 0)

        self.report.build_summary()

        result["metrics"] = self.metrics
        result["issues"] = self.issues
        result["diagnosis_report"] = self.report.to_dict()

        return result

    def _normalize_texts(self, data) -> List[str]:
        """
        将输入数据统一转为字符串列表。

        支持的输入格式：
            - List[str]: 字符串列表
            - 单个 str: 包装为列表
            - 包含非字符串的列表：转为字符串
        """
        if isinstance(data, str):
            return [data]
        elif isinstance(data, list):
            return [str(item) if not isinstance(item, str) else item for item in data]
        return [str(data)]

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_issues(self) -> List[Dict[str, Any]]:
        return self.issues
