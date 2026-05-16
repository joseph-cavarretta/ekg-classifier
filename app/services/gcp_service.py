"""GCP operations orchestration service."""

import logging
from pathlib import Path

from config import Settings
from libs.gcp.bigquery import BigQueryClient
from libs.gcp.storage import GCSClient

logger = logging.getLogger(__name__)


class GCPService:
    """Orchestrates GCP infrastructure and data operations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._storage: GCSClient | None = None
        self._bigquery: BigQueryClient | None = None

    @property
    def storage(self) -> GCSClient:
        if self._storage is None:
            self._storage = GCSClient(self.settings.gcp)
        return self._storage

    @property
    def bigquery(self) -> BigQueryClient:
        if self._bigquery is None:
            self._bigquery = BigQueryClient(self.settings.gcp)
        return self._bigquery

    def setup(self, skip_upload: bool = False) -> None:
        """Set up GCP infrastructure.

        Creates bucket and dataset, optionally uploads data.
        """
        logger.info("Setting up GCP infrastructure")

        self.storage.create_bucket_if_not_exists()
        self.bigquery.create_dataset()

        if not skip_upload:
            self.upload_data(self.settings.data_dir)

    def upload_data(self, data_dir: Path) -> tuple[str, str]:
        """Upload training and test data to GCS.

        Args:
            data_dir: Directory containing data files

        Returns:
            Tuple of (train_uri, test_uri)
        """
        train_path = data_dir / self.settings.train_file
        test_path = data_dir / self.settings.test_file

        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        train_remote = f"electrocardiograms/data/{self.settings.train_file}"
        test_remote = f"electrocardiograms/data/{self.settings.test_file}"

        train_uri = self.storage.upload(train_path, train_remote)
        test_uri = self.storage.upload(test_path, test_remote)

        logger.info(f"Uploaded training data: {train_uri}")
        logger.info(f"Uploaded test data: {test_uri}")

        return train_uri, test_uri

    def load_to_bigquery(self) -> tuple[int, int]:
        """Load data from GCS to BigQuery.

        Returns:
            Tuple of (train_row_count, test_row_count)
        """
        bucket = self.settings.gcp.bucket_name
        dataset = self.settings.gcp.dataset_id
        project = self.settings.gcp.project_id

        train_gcs_uri = f"gs://{bucket}/electrocardiograms/data/{self.settings.train_file}"
        test_gcs_uri = f"gs://{bucket}/electrocardiograms/data/{self.settings.test_file}"

        train_table = f"{project}.{dataset}.processed_train_data"
        test_table = f"{project}.{dataset}.processed_test_data"

        train_rows = self.bigquery.load_from_gcs(
            train_gcs_uri,
            train_table,
            skip_leading_rows=0,
        )

        test_rows = self.bigquery.load_from_gcs(
            test_gcs_uri,
            test_table,
            skip_leading_rows=0,
        )

        return train_rows, test_rows

    def validate_config(self) -> bool:
        """Validate GCP configuration.

        Returns:
            True if configuration is valid
        """
        errors = []

        if not self.settings.gcp.project_id:
            errors.append("GCP_PROJECT_ID is not set")

        if not self.settings.gcp.bucket_name:
            errors.append("GCP_BUCKET_NAME is not set")

        if len(self.settings.gcp.bucket_name) < 3:
            errors.append("Bucket name must be at least 3 characters")

        if errors:
            for error in errors:
                logger.error(error)
            return False

        logger.info("Configuration validated successfully")
        logger.info(f"  Project: {self.settings.gcp.project_id}")
        logger.info(f"  Region: {self.settings.gcp.region}")
        logger.info(f"  Bucket: {self.settings.gcp.bucket_name}")
        logger.info(f"  Dataset: {self.settings.gcp.dataset_id}")

        return True
