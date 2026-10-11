from pathlib import Path

import pytest

from app.services import gcp_service
from app.services.gcp_service import GCPService
from config import GCPConfig, Settings
from errors import MissingFileError


class StubStorage:
    def __init__(self, config: GCPConfig) -> None:
        self.config = config
        self.created = False
        self.uploads: list[tuple[Path, str]] = []

    def create_bucket_if_not_exists(self) -> None:
        self.created = True

    def upload(self, local_path: Path, remote_path: str) -> str:
        self.uploads.append((local_path, remote_path))
        return f"gs://{self.config.bucket_name}/{remote_path}"


class StubBigQuery:
    def __init__(self, config: GCPConfig) -> None:
        self.config = config
        self.created = False
        self.loads: list[tuple[str, str, int]] = []

    def create_dataset(self) -> None:
        self.created = True

    def load_from_gcs(
        self, gcs_uri: str, table_id: str, *, skip_leading_rows: int
    ) -> int:
        self.loads.append((gcs_uri, table_id, skip_leading_rows))
        return 10 * len(self.loads)


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> GCPService:
    monkeypatch.setattr(gcp_service, "GCSClient", StubStorage)
    monkeypatch.setattr(gcp_service, "BigQueryClient", StubBigQuery)
    return GCPService(settings)


def _write_data(settings: Settings) -> None:
    (settings.data_dir / settings.train_file).write_text("x")
    (settings.data_dir / settings.test_file).write_text("x")


def test_clients_are_created_once(service: GCPService) -> None:
    assert service.storage is service.storage
    assert service.bigquery is service.bigquery


def test_setup_creates_resources_and_uploads(
    service: GCPService, settings: Settings
) -> None:
    _write_data(settings)
    service.setup()
    storage = service.storage
    assert isinstance(storage, StubStorage)
    assert storage.created
    assert [remote for _, remote in storage.uploads] == [
        "electrocardiograms/data/mitbih_train.csv.gz",
        "electrocardiograms/data/mitbih_test.csv.gz",
    ]


def test_setup_can_skip_upload(service: GCPService) -> None:
    service.setup(skip_upload=True)
    storage = service.storage
    assert isinstance(storage, StubStorage)
    assert storage.uploads == []


def test_upload_returns_both_uris(service: GCPService, settings: Settings) -> None:
    _write_data(settings)
    train_uri, test_uri = service.upload_data(settings.data_dir)
    assert train_uri.endswith("/mitbih_train.csv.gz")
    assert test_uri.endswith("/mitbih_test.csv.gz")


def test_upload_requires_train_file(service: GCPService, settings: Settings) -> None:
    with pytest.raises(MissingFileError, match="Training file"):
        service.upload_data(settings.data_dir)


def test_upload_requires_test_file(service: GCPService, settings: Settings) -> None:
    (settings.data_dir / settings.train_file).write_text("x")
    with pytest.raises(MissingFileError, match="Test file"):
        service.upload_data(settings.data_dir)


def test_load_to_bigquery_targets_processed_tables(service: GCPService) -> None:
    assert service.load_to_bigquery() == (10, 20)
    bigquery = service.bigquery
    assert isinstance(bigquery, StubBigQuery)
    assert bigquery.loads == [
        (
            "gs://test-bucket/electrocardiograms/data/mitbih_train.csv.gz",
            "test-project.test_dataset.processed_train_data",
            0,
        ),
        (
            "gs://test-bucket/electrocardiograms/data/mitbih_test.csv.gz",
            "test-project.test_dataset.processed_test_data",
            0,
        ),
    ]


def test_validate_config(service: GCPService, settings: Settings) -> None:
    assert service.validate_config() is True

    bad = settings.model_copy(
        update={"gcp": GCPConfig(project_id="", bucket_name="ab")}
    )
    assert GCPService(bad).validate_config() is False
