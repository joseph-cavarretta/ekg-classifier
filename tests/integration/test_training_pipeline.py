"""Integration tests for training pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.training_service import TrainingService
from config import GCPConfig, ModelConfig, Settings


@pytest.fixture
def integration_data(tmp_path: Path) -> tuple[Path, Path]:
    """Create realistic test data files."""
    np.random.seed(42)
    n_features = 187

    # create training data with class imbalance like real data
    train_samples = {
        0: 200,
        1: 50,
        2: 60,
        3: 20,
        4: 70,
    }
    train_data = []
    for label, count in train_samples.items():
        features = np.random.rand(count, n_features)
        labels = np.full((count, 1), float(label))
        train_data.append(np.hstack([features, labels]))

    train_df = pd.DataFrame(np.vstack(train_data))
    train_path = tmp_path / "mitbih_train.csv.gz"
    train_df.to_csv(train_path, index=False, header=False, compression="gzip")

    # create test data
    test_samples = {0: 50, 1: 15, 2: 15, 3: 5, 4: 15}
    test_data = []
    for label, count in test_samples.items():
        features = np.random.rand(count, n_features)
        labels = np.full((count, 1), float(label))
        test_data.append(np.hstack([features, labels]))

    test_df = pd.DataFrame(np.vstack(test_data))
    test_path = tmp_path / "mitbih_test.csv.gz"
    test_df.to_csv(test_path, index=False, header=False, compression="gzip")

    return train_path, test_path


@pytest.fixture
def integration_settings(
    tmp_path: Path, integration_data: tuple[Path, Path]  # noqa: ARG001
) -> Settings:
    """Create settings for integration tests."""
    return Settings(
        data_dir=tmp_path,
        train_file="mitbih_train.csv.gz",
        test_file="mitbih_test.csv.gz",
        output_dir=tmp_path / "output",
        model=ModelConfig(
            hidden_layers=(25, 25),
            max_iter=50,
            random_seed=42,
            balance_classes=True,
            balanced_class_size=50,
        ),
        gcp=GCPConfig(project_id="test", bucket_name="bucket"),
    )


class TestTrainingPipeline:
    def test_sklearn_training_unbalanced(
        self, integration_settings: Settings, tmp_path: Path
    ):
        """Test full sklearn training pipeline without class balancing."""
        integration_settings.model.balance_classes = False

        service = TrainingService(integration_settings)
        result = service.train(
            backend="sklearn",
            output_path=tmp_path / "output" / "model.pkl",
        )

        assert result.accuracy > 0
        assert result.f1_score > 0
        assert result.model_path is not None
        assert result.model_path.exists()

    def test_sklearn_training_balanced(
        self, integration_settings: Settings, tmp_path: Path
    ):
        """Test full sklearn training pipeline with class balancing."""
        service = TrainingService(integration_settings)
        result = service.train(
            backend="sklearn",
            output_path=tmp_path / "output" / "model_balanced.pkl",
        )

        assert result.accuracy > 0
        assert result.f1_score > 0
        assert result.model_path.exists()

    def test_classification_report_format(
        self, integration_settings: Settings
    ):
        """Test that classification report is properly formatted."""
        integration_settings.model.balance_classes = False

        service = TrainingService(integration_settings)
        result = service.train(backend="sklearn")

        assert "precision" in result.classification_report
        assert "recall" in result.classification_report
        assert "f1-score" in result.classification_report

    def test_confusion_matrix_shape(self, integration_settings: Settings):
        """Test that confusion matrix has correct shape."""
        integration_settings.model.balance_classes = False

        service = TrainingService(integration_settings)
        result = service.train(backend="sklearn")

        # 5x5 matrix for 5 classes
        assert len(result.confusion_matrix) == 5
        assert all(len(row) == 5 for row in result.confusion_matrix)

    def test_reproducibility(self, integration_settings: Settings):
        """Test that training is reproducible with same seed."""
        integration_settings.model.balance_classes = False

        service1 = TrainingService(integration_settings)
        result1 = service1.train(backend="sklearn")

        service2 = TrainingService(integration_settings)
        result2 = service2.train(backend="sklearn")

        assert result1.accuracy == result2.accuracy
        assert result1.confusion_matrix == result2.confusion_matrix
