"""Tests for Pydantic models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from models import (
    BigQueryTable,
    DatasetStats,
    GCSUri,
    HeartbeatClass,
    TrainingResult,
)


class TestHeartbeatClass:
    def test_enum_values(self):
        assert HeartbeatClass.NORMAL == 0
        assert HeartbeatClass.SVEB == 1
        assert HeartbeatClass.VEB == 2
        assert HeartbeatClass.FUSION == 3
        assert HeartbeatClass.UNKNOWN == 4

    def test_short_names(self):
        assert HeartbeatClass.NORMAL.short_name == "N"
        assert HeartbeatClass.SVEB.short_name == "S"
        assert HeartbeatClass.VEB.short_name == "V"
        assert HeartbeatClass.FUSION.short_name == "F"
        assert HeartbeatClass.UNKNOWN.short_name == "Q"

    def test_descriptions(self):
        assert "Normal" in HeartbeatClass.NORMAL.description
        assert "Supraventricular" in HeartbeatClass.SVEB.description


class TestDatasetStats:
    def test_valid_stats(self):
        stats = DatasetStats(
            num_samples=1000,
            num_features=187,
            class_distribution={0: 800, 1: 50, 2: 50, 3: 50, 4: 50},
        )
        assert stats.num_samples == 1000
        assert stats.total_classes == 5

    def test_is_balanced_true(self):
        stats = DatasetStats(
            num_samples=500,
            num_features=187,
            class_distribution={0: 100, 1: 100, 2: 100, 3: 100, 4: 100},
        )
        assert stats.is_balanced is True

    def test_is_balanced_false(self):
        stats = DatasetStats(
            num_samples=1000,
            num_features=187,
            class_distribution={0: 800, 1: 50, 2: 50, 3: 50, 4: 50},
        )
        assert stats.is_balanced is False

    def test_negative_samples_rejected(self):
        with pytest.raises(ValidationError):
            DatasetStats(
                num_samples=-1,
                num_features=187,
                class_distribution={},
            )


class TestTrainingResult:
    def test_valid_result(self):
        result = TrainingResult(
            accuracy=0.95,
            f1_score=0.94,
            precision=0.93,
            recall=0.92,
            confusion_matrix=[[90, 10], [5, 95]],
            classification_report="report text",
        )
        assert result.accuracy == 0.95

    def test_summary_property(self):
        result = TrainingResult(
            accuracy=0.95,
            f1_score=0.94,
            precision=0.93,
            recall=0.92,
            confusion_matrix=[[90, 10], [5, 95]],
            classification_report="report",
        )
        summary = result.summary
        assert "0.9500" in summary
        assert "0.9400" in summary

    def test_invalid_accuracy_range(self):
        with pytest.raises(ValidationError):
            TrainingResult(
                accuracy=1.5,
                f1_score=0.94,
                precision=0.93,
                recall=0.92,
                confusion_matrix=[],
                classification_report="",
            )

    def test_model_path_optional(self):
        result = TrainingResult(
            accuracy=0.95,
            f1_score=0.94,
            precision=0.93,
            recall=0.92,
            confusion_matrix=[],
            classification_report="",
        )
        assert result.model_path is None

        result_with_path = TrainingResult(
            accuracy=0.95,
            f1_score=0.94,
            precision=0.93,
            recall=0.92,
            confusion_matrix=[],
            classification_report="",
            model_path=Path("/models/model.pkl"),
        )
        assert result_with_path.model_path == Path("/models/model.pkl")


class TestGCSUri:
    def test_valid_uri(self):
        uri = GCSUri(bucket="my-bucket", path="data/file.csv")
        assert uri.uri == "gs://my-bucket/data/file.csv"

    def test_uri_without_path(self):
        uri = GCSUri(bucket="my-bucket")
        assert uri.uri == "gs://my-bucket"

    def test_with_path_method(self):
        uri = GCSUri(bucket="bucket")
        new_uri = uri.with_path("new/path.csv")
        assert new_uri.uri == "gs://bucket/new/path.csv"
        assert uri.path == ""

    def test_invalid_bucket_name_start(self):
        with pytest.raises(ValidationError):
            GCSUri(bucket="-invalid")

    def test_bucket_name_too_short(self):
        with pytest.raises(ValidationError):
            GCSUri(bucket="ab")


class TestBigQueryTable:
    def test_valid_table(self):
        table = BigQueryTable(
            project_id="my-project",
            dataset_id="my_dataset",
            table_id="my_table",
        )
        assert table.full_table_id == "my-project.my_dataset.my_table"
        assert table.dataset_ref == "my-project.my_dataset"

    def test_project_id_too_short(self):
        with pytest.raises(ValidationError):
            BigQueryTable(
                project_id="short",
                dataset_id="dataset",
                table_id="table",
            )
