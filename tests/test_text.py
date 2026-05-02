import pytest
import numpy as np
from modules.text_detector import (
    TextDetector,
    TextLengthDetector,
    TextDuplicateDetector,
    TextLabelErrorDetector,
    CharacterAnomalyDetector,
    TextLanguageDetector,
    TextPerplexityDetector,
    SentimentConsistencyDetector,
)
from modules.config import Config


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def sample_texts():
    return [
        "This is a good movie with excellent acting",
        "This is a good movie with excellent acting",
        "Bad terrible awful horrible",
        "Hi",
        "A" * 15000,
        "This text has garbled chars",
        "   Leading and trailing spaces   ",
        "This is a normal English text with some content about technology.",
        "Short",
        "Another unique text about science and discovery",
    ]


@pytest.fixture
def sample_labels():
    return [
        "positive", "positive", "negative", "neutral", "neutral",
        "neutral", "neutral", "positive", "neutral", "positive",
    ]


class TestTextLengthDetector:
    def test_basic_detection(self, cfg, sample_texts):
        detector = TextLengthDetector(cfg)
        result = detector.detect(sample_texts)

        assert "issues" in result
        assert "metrics" in result
        assert "length_distribution" in result
        assert result["metrics"]["total_texts"] == len(sample_texts)
        assert result["metrics"]["short_text_count"] >= 1
        assert result["metrics"]["long_text_count"] >= 1

    def test_statistics(self, cfg, sample_texts):
        detector = TextLengthDetector(cfg)
        result = detector.detect(sample_texts)

        stats = result["length_distribution"]["statistics"]
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert stats["min"] >= 0
        assert stats["max"] >= stats["min"]

    def test_empty_input(self, cfg):
        detector = TextLengthDetector(cfg)
        result = detector.detect([])
        assert result["metrics"]["total_texts"] == 0

    def test_histogram(self, cfg, sample_texts):
        detector = TextLengthDetector(cfg)
        result = detector.detect(sample_texts)
        hist = result["length_distribution"]["histogram"]
        assert "bins" in hist
        assert "counts" in hist


class TestTextDuplicateDetector:
    def test_exact_duplicates(self, cfg, sample_texts):
        detector = TextDuplicateDetector(cfg)
        result = detector.detect(sample_texts)

        assert result["metrics"]["exact_duplicate_count"] >= 1
        assert result["metrics"]["unique_text_count"] < len(sample_texts)

    def test_no_duplicates(self, cfg):
        texts = ["unique text one", "unique text two", "unique text three"]
        detector = TextDuplicateDetector(cfg)
        result = detector.detect(texts)

        assert result["metrics"]["exact_duplicate_count"] == 0
        assert result["metrics"]["unique_text_count"] == 3

    def test_empty_input(self, cfg):
        detector = TextDuplicateDetector(cfg)
        result = detector.detect([])
        assert result["metrics"]["total_texts"] == 0


class TestTextLabelErrorDetector:
    def test_label_error_detection(self, cfg):
        texts = [
            "This movie is great and wonderful",
            "I love this amazing film",
            "Terrible awful horrible movie",
            "Bad acting poor script",
            "The weather is nice today",
            "I enjoy reading books",
            "This product is excellent",
            "Disgusting waste of money",
            "Beautiful scenery and great music",
            "Worst experience ever had",
        ]
        labels = [
            "positive", "positive", "negative", "negative", "positive",
            "negative", "positive", "positive", "negative", "negative",
        ]

        detector = TextLabelErrorDetector(cfg)
        result = detector.detect(texts, labels)

        assert "error_count" in result
        assert "error_rate" in result
        assert "label_quality_scores" in result
        assert "class_names" in result
        assert len(result["label_quality_scores"]) == len(texts)

    def test_insufficient_samples(self, cfg):
        texts = ["one text", "another text"]
        labels = ["A", "A"]

        detector = TextLabelErrorDetector(cfg)
        result = detector.detect(texts, labels)

        assert "error" in result or "error_count" in result


class TestCharacterAnomalyDetector:
    def test_garbled_chars(self, cfg):
        texts = ["Normal text", "Text with \ufffd replacement char"]
        detector = CharacterAnomalyDetector(cfg)
        result = detector.detect(texts)

        assert result["metrics"]["garbled_text_count"] >= 1

    def test_whitespace_padding(self, cfg):
        texts = ["Normal text", "   padded text   "]
        detector = CharacterAnomalyDetector(cfg)
        result = detector.detect(texts)

        assert result["metrics"]["whitespace_padding_count"] >= 1

    def test_high_special_chars(self, cfg):
        texts = ["Normal text", "!@#$%^&*()!@#$%^&*()!@#$%^&*()"]
        detector = CharacterAnomalyDetector(cfg)
        result = detector.detect(texts)

        assert result["metrics"]["high_special_char_count"] >= 1

    def test_clean_text(self, cfg):
        texts = ["This is clean normal text", "Another clean text here"]
        detector = CharacterAnomalyDetector(cfg)
        result = detector.detect(texts)

        assert result["metrics"]["garbled_text_count"] == 0


class TestTextLanguageDetector:
    def test_english_detection(self, cfg):
        texts = ["This is English text", "Another English sentence"]
        detector = TextLanguageDetector(cfg)
        result = detector.detect(texts)

        assert "language_distribution" in result["metrics"]

    def test_mixed_language(self, cfg):
        texts = ["This is English with some Chinese characters mixed in"]
        detector = TextLanguageDetector(cfg)
        result = detector.detect(texts)

        assert "metrics" in result


class TestTextPerplexityDetector:
    def test_basic_perplexity(self, cfg, sample_texts):
        detector = TextPerplexityDetector(cfg)
        result = detector.detect(sample_texts)

        assert "perplexity_scores" in result
        assert "metrics" in result
        assert result["method"] == "statistical_approximation"

    def test_empty_input(self, cfg):
        detector = TextPerplexityDetector(cfg)
        result = detector.detect([])

        assert result["metrics"]["total_texts"] == 0


class TestSentimentConsistencyDetector:
    def test_inconsistency_detection(self, cfg):
        texts = [
            "This is great wonderful amazing",
            "Terrible awful horrible bad",
        ]
        labels = ["negative", "positive"]

        detector = SentimentConsistencyDetector(cfg)
        result = detector.detect(texts, labels)

        assert result["metrics"]["inconsistency_count"] >= 1

    def test_consistent_labels(self, cfg):
        texts = [
            "This is great wonderful amazing",
            "Terrible awful horrible bad",
        ]
        labels = ["positive", "negative"]

        detector = SentimentConsistencyDetector(cfg)
        result = detector.detect(texts, labels)

        assert result["metrics"]["inconsistency_count"] == 0


class TestTextDetector:
    def test_full_detection(self, cfg, sample_texts, sample_labels):
        detector = TextDetector(cfg)
        result = detector.detect(
            sample_texts,
            labels=sample_labels,
            modules=["text_length", "duplicate", "character_anomaly"],
        )

        assert "modules_run" in result
        assert "metrics" in result
        assert "issues" in result
        assert result["metrics"]["total_texts"] == len(sample_texts)

    def test_default_modules(self, cfg, sample_texts):
        detector = TextDetector(cfg)
        result = detector.detect(sample_texts)

        assert "text_length" in result["modules_run"]
        assert "duplicate" in result["modules_run"]
        assert "character_anomaly" in result["modules_run"]

    def test_custom_modules(self, cfg, sample_texts):
        detector = TextDetector(cfg)
        result = detector.detect(sample_texts, modules=["text_length"])

        assert result["modules_run"] == ["text_length"]
        assert result["text_length"] is not None
        assert result["duplicate"] is None

    def test_modality(self, cfg):
        detector = TextDetector(cfg)
        assert detector.modality() == "text"

    def test_normalize_texts(self, cfg):
        detector = TextDetector(cfg)
        assert detector._normalize_texts("single text") == ["single text"]
        assert detector._normalize_texts(["a", "b"]) == ["a", "b"]
        assert detector._normalize_texts([1, 2]) == ["1", "2"]

    def test_get_metrics_and_issues(self, cfg, sample_texts):
        detector = TextDetector(cfg)
        detector.detect(sample_texts, modules=["text_length"])

        metrics = detector.get_metrics()
        issues = detector.get_issues()

        assert isinstance(metrics, dict)
        assert isinstance(issues, list)
