"""Tests for data loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import GCPConfig, Settings
from libs.data.loader import LocalDataLoader


@pytest.fixture
def temp_data_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary CSV files for testing."""
    np.random.seed(42)
    n_features = 187

    # create train data
    train_data = np.random.rand(100, n_features + 1).astype(np.float64)
    train_data[:, -1] = np.repeat([0, 1, 2, 3, 4], 20)
    train_df = pd.DataFrame(train_data)
    train_path = tmp_path / "train.csv"
    train_df.to_csv(train_path, index=False, header=False)

    # create test data
    test_data = np.random.rand(50, n_features + 1).astype(np.float64)
    test_data[:, -1] = np.repeat([0, 1, 2, 3, 4], 10)
    test_df = pd.DataFrame(test_data)
    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False, header=False)

    return train_path, test_path


@pytest.fixture
def loader_settings(tmp_path: Path, temp_data_files: tuple[Path, Path]) -> Settings:
    """Create settings pointing to temp data files."""
    train_path, test_path = temp_data_files
    return Settings(
        data_dir=tmp_path,
        train_file=train_path.name,
        test_file=test_path.name,
        gcp=GCPConfig(project_id="test", bucket_name="bucket"),
    )


class TestLocalDataLoader:
    def test_load_train(self, loader_settings: Settings):
        loader = LocalDataLoader(loader_settings)
        data = loader.load_train()

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert len(data.columns) == 188

    def test_load_test(self, loader_settings: Settings):
        loader = LocalDataLoader(loader_settings)
        data = loader.load_test()

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50

    def test_dtype_optimization(self, loader_settings: Settings):
        loader = LocalDataLoader(loader_settings)
        data = loader.load_train()

        # all columns should be float32 after optimization
        for col in data.columns:
            assert data[col].dtype == np.float32

    def test_get_stats(self, loader_settings: Settings):
        loader = LocalDataLoader(loader_settings)
        data = loader.load_train()
        stats = loader.get_stats(data)

        assert stats.num_samples == 100
        assert stats.num_features == 187
        assert stats.total_classes == 5
        assert sum(stats.class_distribution.values()) == 100

    def test_file_not_found(self, tmp_path: Path):
        settings = Settings(
            data_dir=tmp_path,
            train_file="nonexistent.csv",
            gcp=GCPConfig(project_id="test", bucket_name="bucket"),
        )
        loader = LocalDataLoader(settings)

        with pytest.raises(FileNotFoundError):
            loader.load_train()

    def test_null_values_rejected(self, tmp_path: Path):
        # create data with null values
        data = pd.DataFrame(
            {
                0: [1.0, 2.0, None],
                1: [1.0, 2.0, 3.0],
            }
        )
        data_path = tmp_path / "null_data.csv"
        data.to_csv(data_path, index=False, header=False)

        settings = Settings(
            data_dir=tmp_path,
            train_file="null_data.csv",
            gcp=GCPConfig(project_id="test", bucket_name="bucket"),
        )
        loader = LocalDataLoader(settings)

        with pytest.raises(ValueError, match="Null values"):
            loader.load_train()
