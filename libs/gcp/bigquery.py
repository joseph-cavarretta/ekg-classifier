import logging

import pandas as pd
from google.api_core.exceptions import Conflict, NotFound
from google.cloud import bigquery

from config import GCPConfig

logger = logging.getLogger(__name__)


class BigQueryClient:
    """BigQuery client with error handling."""

    def __init__(self, config: GCPConfig) -> None:
        self.config = config
        self._client = bigquery.Client(project=config.project_id)

    def create_dataset(
        self, dataset_id: str | None = None, location: str = "US"
    ) -> None:
        """Create a dataset if it doesn't exist.

        Args:
            dataset_id: Dataset ID (uses config default if not provided)
            location: Dataset location
        """
        dataset_id = dataset_id or self.config.dataset_id
        full_dataset_id = f"{self.config.project_id}.{dataset_id}"

        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = location

        try:
            self._client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset: {full_dataset_id}")
        except Conflict:
            logger.info(f"Dataset already exists: {full_dataset_id}")

    def load_from_gcs(
        self,
        gcs_uri: str,
        table_id: str,
        *,
        autodetect: bool = True,
        skip_leading_rows: int = 1,
        write_disposition: str = "WRITE_TRUNCATE",
    ) -> int:
        """Load data from GCS into BigQuery.

        Args:
            gcs_uri: Full GCS URI (gs://bucket/path)
            table_id: Full table ID (project.dataset.table)
            autodetect: Auto-detect schema
            skip_leading_rows: Number of header rows to skip
            write_disposition: How to handle existing data

        Returns:
            Number of rows loaded
        """
        job_config = bigquery.LoadJobConfig(
            autodetect=autodetect,
            source_format=bigquery.SourceFormat.CSV,
            create_disposition="CREATE_IF_NEEDED",
            write_disposition=write_disposition,
            skip_leading_rows=skip_leading_rows,
        )

        logger.info(f"Loading {gcs_uri} to {table_id}")

        load_job = self._client.load_table_from_uri(
            gcs_uri,
            table_id,
            job_config=job_config,
        )
        load_job.result()

        table = self._client.get_table(table_id)
        row_count = table.num_rows or 0

        logger.info(f"Loaded {row_count} rows to {table_id}")
        return row_count

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame.

        Args:
            sql: SQL query to execute

        Returns:
            Query results as pandas DataFrame
        """
        logger.debug(f"Executing query: {sql[:100]}...")
        query_job = self._client.query(sql)
        return query_job.to_dataframe()

    def table_exists(self, table_id: str) -> bool:
        """Check if a table exists."""
        try:
            self._client.get_table(table_id)
            return True
        except NotFound:
            return False

    def get_table_row_count(self, table_id: str) -> int:
        """Get the row count of a table."""
        table = self._client.get_table(table_id)
        return table.num_rows or 0
