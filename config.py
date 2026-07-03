from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GCPConfig(BaseSettings):
    """Google Cloud Platform configuration."""

    model_config = SettingsConfigDict(env_prefix="GCP_")

    project_id: str = Field(description="GCP project ID")
    region: str = Field(default="us-central1", description="GCP region")
    bucket_name: str = Field(description="GCS bucket name for data and models")
    dataset_id: str = Field(
        default="electrocardiograms", description="BigQuery dataset ID"
    )


class DataprocConfig(BaseSettings):
    """Dataproc cluster configuration."""

    model_config = SettingsConfigDict(env_prefix="DATAPROC_")

    cluster_name: str = Field(
        default="ekg-classifier-cluster", description="Dataproc cluster name"
    )
    num_workers: int = Field(default=2, description="Number of worker nodes")
    master_machine_type: str = Field(
        default="n1-standard-2", description="Master node machine type"
    )
    worker_machine_type: str = Field(
        default="n1-standard-2", description="Worker node machine type"
    )
    master_boot_disk_gb: int = Field(
        default=30, description="Master boot disk size in GB"
    )
    worker_boot_disk_gb: int = Field(
        default=30, description="Worker boot disk size in GB"
    )
    image_version: str = Field(
        default="2.1-debian11", description="Dataproc image version"
    )
    enable_jupyter: bool = Field(
        default=True, description="Enable Jupyter component on cluster"
    )


class ModelConfig(BaseSettings):
    """Neural network model configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    hidden_layers: tuple[int, ...] = Field(
        default=(75, 75, 75), description="Hidden layer sizes for MLP"
    )
    max_iter: int = Field(default=300, description="Maximum training iterations")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    block_size: int = Field(default=128, description="Block size for Spark MLP")
    balance_classes: bool = Field(
        default=True, description="Balance classes during training"
    )
    balanced_class_size: int = Field(
        default=18000, description="Target size per class when balancing"
    )


class Settings(BaseSettings):
    """Application settings with nested configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data_dir: Path = Field(
        default=Path("data"), description="Directory containing data files"
    )
    train_file: str = Field(
        default="mitbih_train.csv.gz", description="Training data filename"
    )
    test_file: str = Field(
        default="mitbih_test.csv.gz", description="Test data filename"
    )
    output_dir: Path = Field(
        default=Path("output"), description="Directory for model outputs"
    )

    gcp: GCPConfig = Field(
        default_factory=lambda: GCPConfig(
            project_id="",
            bucket_name="",
        )
    )
    dataproc: DataprocConfig = Field(default_factory=DataprocConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file


def get_settings() -> Settings:
    """Load settings from environment variables and .env file."""
    return Settings()
