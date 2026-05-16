"""Tests for configuration module."""

from pathlib import Path

from config import GCPConfig, ModelConfig, Settings


class TestGCPConfig:
    def test_default_values(self):
        config = GCPConfig(project_id="test", bucket_name="bucket")
        assert config.region == "us-central1"
        assert config.dataset_id == "electrocardiograms"

    def test_custom_values(self):
        config = GCPConfig(
            project_id="my-project",
            bucket_name="my-bucket",
            region="europe-west1",
            dataset_id="custom_dataset",
        )
        assert config.project_id == "my-project"
        assert config.bucket_name == "my-bucket"
        assert config.region == "europe-west1"
        assert config.dataset_id == "custom_dataset"


class TestModelConfig:
    def test_default_values(self):
        config = ModelConfig()
        assert config.hidden_layers == (75, 75, 75)
        assert config.max_iter == 300
        assert config.random_seed == 42
        assert config.balance_classes is True

    def test_custom_hidden_layers(self):
        config = ModelConfig(hidden_layers=(100, 100))
        assert config.hidden_layers == (100, 100)


class TestSettings:
    def test_default_paths(self):
        settings = Settings(
            gcp=GCPConfig(project_id="test", bucket_name="bucket")
        )
        assert settings.data_dir == Path("data")
        assert settings.train_file == "mitbih_train.csv.gz"
        assert settings.test_file == "mitbih_test.csv.gz"

    def test_train_path_property(self):
        settings = Settings(
            data_dir=Path("/custom/path"),
            gcp=GCPConfig(project_id="test", bucket_name="bucket"),
        )
        assert settings.train_path == Path("/custom/path/mitbih_train.csv.gz")

    def test_test_path_property(self):
        settings = Settings(
            data_dir=Path("/custom/path"),
            gcp=GCPConfig(project_id="test", bucket_name="bucket"),
        )
        assert settings.test_path == Path("/custom/path/mitbih_test.csv.gz")

    def test_nested_config_access(self):
        settings = Settings(
            gcp=GCPConfig(project_id="test", bucket_name="bucket"),
            model=ModelConfig(max_iter=500),
        )
        assert settings.gcp.project_id == "test"
        assert settings.model.max_iter == 500
