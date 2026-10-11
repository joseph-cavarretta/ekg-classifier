import logging
from pathlib import Path

from google.api_core.exceptions import Conflict, NotFound
from google.cloud import storage  # type: ignore[attr-defined]

from config import GCPConfig
from errors import MissingFileError

logger = logging.getLogger(__name__)


class GCSClient:
    """Google Cloud Storage client with error handling."""

    def __init__(self, config: GCPConfig) -> None:
        self.config = config
        self._client = storage.Client(project=config.project_id)

    @property
    def bucket(self) -> storage.Bucket:
        """The configured bucket (no API call until it is used)."""
        return self._client.bucket(self.config.bucket_name)

    def create_bucket_if_not_exists(self, location: str = "US") -> None:
        """Create the storage bucket if it doesn't exist."""
        try:
            self._client.create_bucket(
                self.config.bucket_name,
                location=location,
            )
            logger.info("Created bucket: %s", self.config.bucket_name)
        except Conflict:
            logger.info("Bucket already exists: %s", self.config.bucket_name)

    def upload(self, local_path: Path, remote_path: str) -> str:
        """Upload local_path to remote_path (no gs:// prefix); return its gs:// URI."""
        if not local_path.exists():
            raise MissingFileError(f"Local file not found: {local_path}")

        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(str(local_path))

        uri = f"gs://{self.config.bucket_name}/{remote_path}"
        logger.info("Uploaded %s to %s", local_path, uri)
        return uri

    def download(self, remote_path: str, local_path: Path) -> Path:
        """Download remote_path (no gs:// prefix) to local_path and return local_path.

        Raises NotFound when the blob does not exist.
        """
        blob = self.bucket.blob(remote_path)

        if not blob.exists():
            raise NotFound(  # type: ignore[no-untyped-call]
                f"Remote file not found: gs://{self.config.bucket_name}/{remote_path}"
            )

        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

        logger.info(
            "Downloaded gs://%s/%s to %s",
            self.config.bucket_name,
            remote_path,
            local_path,
        )
        return local_path

    def exists(self, remote_path: str) -> bool:
        """Check if a remote path exists in the bucket."""
        exists: bool = self.bucket.blob(remote_path).exists()
        return exists

    def list_blobs(self, prefix: str = "") -> list[str]:
        """List all blob names with the given prefix."""
        blobs = self._client.list_blobs(self.config.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs]
