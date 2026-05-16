import logging
from pathlib import Path

import pandas as pd

from config import Settings
from models import DatasetStats

logger = logging.getLogger(__name__)


class LocalDataLoader:
    """Load EKG data from local CSV files."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_train(self) -> pd.DataFrame:
        """Load training dataset with dtype optimization."""
        logger.info(f"Loading training data from {self.settings.train_path}")
        return self._load_and_optimize(self.settings.train_path)

    def load_test(self) -> pd.DataFrame:
        """Load test dataset with dtype optimization."""
        logger.info(f"Loading test data from {self.settings.test_path}")
        return self._load_and_optimize(self.settings.test_path)

    def _load_and_optimize(self, path: Path) -> pd.DataFrame:
        """Load CSV and convert float64 to float32 to reduce memory usage."""
        compression = "gzip" if path.suffix == ".gz" else None
        df = pd.read_csv(path, header=None, compression=compression)

        for col in df.columns:
            if df[col].dtype == "float64":
                df[col] = pd.to_numeric(df[col], downcast="float")

        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        self._validate_no_nulls(df, path.name)
        return df

    def _validate_no_nulls(self, df: pd.DataFrame, name: str) -> None:
        """Ensure no null values in dataset."""
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            raise ValueError(f"Null values found in {name} columns: {null_cols}")

    def get_stats(self, data: pd.DataFrame) -> DatasetStats:
        """Compute statistics for a dataset."""
        label_col = data.columns[-1]
        class_counts = data[label_col].value_counts().to_dict()
        class_distribution = {int(k): int(v) for k, v in class_counts.items()}

        return DatasetStats(
            num_samples=len(data),
            num_features=len(data.columns) - 1,
            class_distribution=class_distribution,
        )
