"""Shared test fixtures."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from config import GCPConfig, ModelConfig, Settings


@pytest.fixture
def sample_train_data() -> pd.DataFrame:
    """Create sample training data with 5 classes."""
    np.random.seed(42)
    n_samples = 500
    n_features = 187

    # generate random features between 0 and 1
    features = np.random.rand(n_samples, n_features).astype(np.float32)

    # generate labels (100 samples per class)
    labels = np.repeat([0.0, 1.0, 2.0, 3.0, 4.0], 100).astype(np.float32)
    np.random.shuffle(labels)

    data = np.column_stack([features, labels])
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data() -> pd.DataFrame:
    """Create sample test data with 5 classes."""
    np.random.seed(43)
    n_samples = 100
    n_features = 187

    features = np.random.rand(n_samples, n_features).astype(np.float32)
    labels = np.repeat([0.0, 1.0, 2.0, 3.0, 4.0], 20).astype(np.float32)
    np.random.shuffle(labels)

    data = np.column_stack([features, labels])
    return pd.DataFrame(data)


@pytest.fixture
def model_config() -> ModelConfig:
    """Create test model configuration."""
    return ModelConfig(
        hidden_layers=(25, 25),
        max_iter=50,
        random_seed=42,
        balance_classes=False,
    )


@pytest.fixture
def gcp_config() -> GCPConfig:
    """Create test GCP configuration."""
    return GCPConfig(
        project_id="test-project",
        bucket_name="test-bucket",
        region="us-central1",
        dataset_id="test_dataset",
    )


@pytest.fixture
def settings(
    tmp_path: Path, model_config: ModelConfig, gcp_config: GCPConfig
) -> Settings:
    """Create test settings with temp directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    return Settings(
        data_dir=data_dir,
        output_dir=tmp_path / "output",
        model=model_config,
        gcp=gcp_config,
    )


@pytest.fixture
def mock_gcs_client() -> MagicMock:
    """Create a mock GCS client."""
    client = MagicMock()
    client.create_bucket_if_not_exists.return_value = None
    client.upload.return_value = "gs://test-bucket/path/file.csv"
    client.download.return_value = Path("/tmp/downloaded.csv")
    client.exists.return_value = True
    return client


@pytest.fixture
def mock_bigquery_client() -> MagicMock:
    """Create a mock BigQuery client."""
    client = MagicMock()
    client.create_dataset.return_value = None
    client.load_from_gcs.return_value = 1000
    client.table_exists.return_value = True
    return client
