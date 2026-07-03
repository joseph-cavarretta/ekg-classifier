import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.api_core.exceptions import Conflict, NotFound
from google.cloud import storage  # type: ignore[attr-defined]

from config import GCPConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GCSClient:
    """Google Cloud Storage client with error handling."""

    def __init__(self, config: GCPConfig) -> None:
        self.config = config
        self._client: Any = storage.Client(project=config.project_id)

    @property
    def bucket(self) -> Any:
        return self._client.bucket(self.config.bucket_name)

    def create_bucket_if_not_exists(self, location: str = "US") -> None:
        """Create the storage bucket if it doesn't exist."""
        try:
            self._client.create_bucket(
                self.config.bucket_name,
                location=location,
            )
            logger.info(f"Created bucket: {self.config.bucket_name}")
        except Conflict:
            logger.info(f"Bucket already exists: {self.config.bucket_name}")

    def upload(self, local_path: Path, remote_path: str) -> str:
        """Upload a file to GCS.

        Args:
            local_path: Local file path to upload
            remote_path: Destination path in the bucket (without gs:// prefix)

        Returns:
            The full GCS URI of the uploaded file
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(str(local_path))

        uri = f"gs://{self.config.bucket_name}/{remote_path}"
        logger.info(f"Uploaded {local_path} to {uri}")
        return uri

    def download(self, remote_path: str, local_path: Path) -> Path:
        """Download a file from GCS.

        Args:
            remote_path: Source path in the bucket (without gs:// prefix)
            local_path: Local destination path

        Returns:
            The local path of the downloaded file
        """
        blob = self.bucket.blob(remote_path)

        if not blob.exists():
            raise NotFound(  # type: ignore[no-untyped-call]
                f"Remote file not found: gs://{self.config.bucket_name}/{remote_path}"
            )

        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

        logger.info(
            f"Downloaded gs://{self.config.bucket_name}/{remote_path} to {local_path}"
        )
        return local_path

    def exists(self, remote_path: str) -> bool:
        """Check if a remote path exists in the bucket."""
        blob = self.bucket.blob(remote_path)
        return blob.exists()

    def list_blobs(self, prefix: str = "") -> list[str]:
        """List all blob names with the given prefix."""
        blobs = self._client.list_blobs(self.config.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs]
