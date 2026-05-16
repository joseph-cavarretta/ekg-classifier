from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from models import DatasetStats, TrainingResult


class DataLoader(Protocol):
    """Protocol for loading training and test data."""

    def load_train(self) -> pd.DataFrame:
        """Load training dataset."""
        ...

    def load_test(self) -> pd.DataFrame:
        """Load test dataset."""
        ...

    def get_stats(self, data: pd.DataFrame) -> DatasetStats:
        """Compute statistics for a dataset."""
        ...


class ModelTrainer(Protocol):
    """Protocol for training ML models."""

    def train(self, x_train: Any, y_train: Any) -> None:
        """Train the model on provided data."""
        ...

    def predict(self, x: Any) -> Any:
        """Generate predictions for input data."""
        ...

    def evaluate(self, x_test: Any, y_test: Any) -> TrainingResult:
        """Evaluate model on test data and return metrics."""
        ...

    def save(self, path: Path) -> None:
        """Save trained model to disk."""
        ...

    def load(self, path: Path) -> None:
        """Load trained model from disk."""
        ...


class CloudStorage(Protocol):
    """Protocol for cloud storage operations."""

    def upload(self, local_path: Path, remote_path: str) -> str:
        """Upload a file to cloud storage. Returns the remote URI."""
        ...

    def download(self, remote_path: str, local_path: Path) -> Path:
        """Download a file from cloud storage. Returns the local path."""
        ...

    def exists(self, remote_path: str) -> bool:
        """Check if a remote path exists."""
        ...

    def create_bucket_if_not_exists(self) -> None:
        """Create the storage bucket if it doesn't exist."""
        ...


class BigQueryClientProtocol(Protocol):
    """Protocol for BigQuery operations."""

    def create_dataset(self, dataset_id: str, location: str = "US") -> None:
        """Create a dataset if it doesn't exist."""
        ...

    def load_from_gcs(
        self,
        gcs_uri: str,
        table_id: str,
        *,
        autodetect: bool = True,
        skip_leading_rows: int = 1,
    ) -> int:
        """Load data from GCS into BigQuery. Returns row count."""
        ...

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        ...
