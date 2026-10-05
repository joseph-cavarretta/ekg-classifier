import logging
from pathlib import Path

from config import Settings
from errors import MissingFileError
from libs.gcp.bigquery import BigQueryClient
from libs.gcp.storage import GCSClient

logger = logging.getLogger(__name__)

# GCS rejects bucket names shorter than this.
MIN_BUCKET_NAME_LENGTH = 3


class GCPService:
    """Orchestrates GCP infrastructure and data operations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._storage: GCSClient | None = None
        self._bigquery: BigQueryClient | None = None

    @property
    def storage(self) -> GCSClient:
        """The GCS client, created on first use."""
        if self._storage is None:
            self._storage = GCSClient(self.settings.gcp)
        return self._storage

    @property
    def bigquery(self) -> BigQueryClient:
        """The BigQuery client, created on first use."""
        if self._bigquery is None:
            self._bigquery = BigQueryClient(self.settings.gcp)
        return self._bigquery

    def setup(self, skip_upload: bool = False) -> None:
        """Create the bucket and dataset, then upload the data unless skip_upload."""
        logger.info("Setting up GCP infrastructure")

        self.storage.create_bucket_if_not_exists()
        self.bigquery.create_dataset()

        if not skip_upload:
            self.upload_data(self.settings.data_dir)

    def upload_data(self, data_dir: Path) -> tuple[str, str]:
        """Upload the train and test files in data_dir; return (train_uri, test_uri)."""
        train_path = data_dir / self.settings.train_file
        test_path = data_dir / self.settings.test_file

        if not train_path.exists():
            raise MissingFileError(f"Training file not found: {train_path}")
        if not test_path.exists():
            raise MissingFileError(f"Test file not found: {test_path}")

        train_remote = f"electrocardiograms/data/{self.settings.train_file}"
        test_remote = f"electrocardiograms/data/{self.settings.test_file}"

        train_uri = self.storage.upload(train_path, train_remote)
        test_uri = self.storage.upload(test_path, test_remote)

        logger.info("Uploaded training data: %s", train_uri)
        logger.info("Uploaded test data: %s", test_uri)

        return train_uri, test_uri

    def load_to_bigquery(self) -> tuple[int, int]:
        """Load the uploaded files into BigQuery; return (train_rows, test_rows)."""
        bucket = self.settings.gcp.bucket_name
        dataset = self.settings.gcp.dataset_id
        project = self.settings.gcp.project_id

        train_gcs_uri = (
            f"gs://{bucket}/electrocardiograms/data/{self.settings.train_file}"
        )
        test_gcs_uri = (
            f"gs://{bucket}/electrocardiograms/data/{self.settings.test_file}"
        )

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
        """Log every GCP configuration problem; return True when there are none."""
        errors = []

        if not self.settings.gcp.project_id:
            errors.append("GCP_PROJECT_ID is not set")

        if not self.settings.gcp.bucket_name:
            errors.append("GCP_BUCKET_NAME is not set")

        if len(self.settings.gcp.bucket_name) < MIN_BUCKET_NAME_LENGTH:
            errors.append("Bucket name must be at least 3 characters")

        if errors:
            for error in errors:
                logger.error(error)
            return False

        logger.info("Configuration validated successfully")
        logger.info("  Project: %s", self.settings.gcp.project_id)
        logger.info("  Region: %s", self.settings.gcp.region)
        logger.info("  Bucket: %s", self.settings.gcp.bucket_name)
        logger.info("  Dataset: %s", self.settings.gcp.dataset_id)

        return True
