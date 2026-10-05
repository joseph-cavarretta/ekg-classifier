from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import Conflict, NotFound
from google.cloud import dataproc_v1

from config import DataprocConfig, GCPConfig
from errors import MissingFileError
from libs.gcp.bigquery import BigQueryClient
from libs.gcp.dataproc import SPARK_BIGQUERY_JAR, DataprocClient
from libs.gcp.storage import GCSClient

# The google clients open network connections on construction, so each test swaps
# in a MagicMock for the vendor client class; that is the only seam they offer.


def _conflict() -> Conflict:
    return Conflict("exists")  # type: ignore[no-untyped-call]  # untyped in google stubs


def _not_found() -> NotFound:
    return NotFound("missing")  # type: ignore[no-untyped-call]  # untyped in google stubs


@pytest.fixture
def gcs(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    monkeypatch.setattr("google.cloud.storage.Client", lambda **_: client)
    return client


@pytest.fixture
def bq(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    monkeypatch.setattr("google.cloud.bigquery.Client", lambda **_: client)
    return client


class TestGCSClient:
    def test_create_bucket(self, gcs: MagicMock, gcp_config: GCPConfig) -> None:
        GCSClient(gcp_config).create_bucket_if_not_exists(location="EU")
        gcs.create_bucket.assert_called_once_with("test-bucket", location="EU")

    def test_existing_bucket_is_not_an_error(
        self, gcs: MagicMock, gcp_config: GCPConfig
    ) -> None:
        gcs.create_bucket.side_effect = _conflict()
        GCSClient(gcp_config).create_bucket_if_not_exists()

    def test_upload_returns_uri(
        self, gcs: MagicMock, gcp_config: GCPConfig, tmp_path: Path
    ) -> None:
        local = tmp_path / "train.csv"
        local.write_text("x")

        uri = GCSClient(gcp_config).upload(local, "data/train.csv")

        assert uri == "gs://test-bucket/data/train.csv"
        blob = gcs.bucket.return_value.blob
        blob.assert_called_once_with("data/train.csv")
        blob.return_value.upload_from_filename.assert_called_once_with(str(local))

    @pytest.mark.usefixtures("gcs")
    def test_upload_missing_file_raises(
        self, gcp_config: GCPConfig, tmp_path: Path
    ) -> None:
        with pytest.raises(MissingFileError):
            GCSClient(gcp_config).upload(tmp_path / "nope.csv", "x")

    def test_download_writes_into_new_directory(
        self, gcs: MagicMock, gcp_config: GCPConfig, tmp_path: Path
    ) -> None:
        target = tmp_path / "nested" / "file.csv"
        blob = gcs.bucket.return_value.blob.return_value
        blob.exists.return_value = True

        assert GCSClient(gcp_config).download("data/file.csv", target) == target
        assert target.parent.is_dir()
        blob.download_to_filename.assert_called_once_with(str(target))

    def test_download_missing_blob_raises(
        self, gcs: MagicMock, gcp_config: GCPConfig, tmp_path: Path
    ) -> None:
        gcs.bucket.return_value.blob.return_value.exists.return_value = False
        with pytest.raises(NotFound):
            GCSClient(gcp_config).download("data/file.csv", tmp_path / "f.csv")

    def test_exists_and_list(self, gcs: MagicMock, gcp_config: GCPConfig) -> None:
        gcs.bucket.return_value.blob.return_value.exists.return_value = False
        first, second = MagicMock(), MagicMock()
        first.name, second.name = "a.csv", "b.csv"
        gcs.list_blobs.return_value = [first, second]
        client = GCSClient(gcp_config)

        assert client.exists("a.csv") is False
        assert client.list_blobs("data/") == ["a.csv", "b.csv"]
        gcs.list_blobs.assert_called_once_with("test-bucket", prefix="data/")


class TestBigQueryClient:
    def test_create_dataset_defaults_to_config(
        self, bq: MagicMock, gcp_config: GCPConfig
    ) -> None:
        BigQueryClient(gcp_config).create_dataset()
        dataset = bq.create_dataset.call_args.args[0]
        assert dataset.dataset_id == "test_dataset"
        assert dataset.location == "US"

    def test_existing_dataset_is_not_an_error(
        self, bq: MagicMock, gcp_config: GCPConfig
    ) -> None:
        bq.create_dataset.side_effect = _conflict()
        BigQueryClient(gcp_config).create_dataset("other")

    def test_load_from_gcs_returns_row_count(
        self, bq: MagicMock, gcp_config: GCPConfig
    ) -> None:
        bq.get_table.return_value.num_rows = 42

        rows = BigQueryClient(gcp_config).load_from_gcs(
            "gs://b/f.csv", "p.d.t", skip_leading_rows=0
        )

        assert rows == 42
        job_config = bq.load_table_from_uri.call_args.kwargs["job_config"]
        assert job_config.skip_leading_rows == 0
        bq.load_table_from_uri.return_value.result.assert_called_once()

    def test_query_returns_dataframe(
        self, bq: MagicMock, gcp_config: GCPConfig
    ) -> None:
        frame = object()
        bq.query.return_value.to_dataframe.return_value = frame
        assert BigQueryClient(gcp_config).query("SELECT 1") is frame

    def test_table_exists(self, bq: MagicMock, gcp_config: GCPConfig) -> None:
        client = BigQueryClient(gcp_config)
        assert client.table_exists("p.d.t") is True
        bq.get_table.side_effect = _not_found()
        assert client.table_exists("p.d.t") is False

    def test_table_exists_raises_other_errors(
        self, bq: MagicMock, gcp_config: GCPConfig
    ) -> None:
        bq.get_table.side_effect = PermissionError("denied")
        with pytest.raises(PermissionError):
            BigQueryClient(gcp_config).table_exists("p.d.t")

    def test_row_count_of_empty_table_is_zero(
        self, bq: MagicMock, gcp_config: GCPConfig
    ) -> None:
        bq.get_table.return_value.num_rows = None
        assert BigQueryClient(gcp_config).get_table_row_count("p.d.t") == 0


@pytest.fixture
def dataproc(
    monkeypatch: pytest.MonkeyPatch, gcp_config: GCPConfig
) -> tuple[DataprocClient, MagicMock, MagicMock]:
    clusters, jobs = MagicMock(), MagicMock()
    monkeypatch.setattr(
        "google.cloud.dataproc_v1.ClusterControllerClient",
        lambda **_: clusters,
    )
    monkeypatch.setattr(
        "google.cloud.dataproc_v1.JobControllerClient", lambda **_: jobs
    )
    client = DataprocClient(gcp_config, DataprocConfig(num_workers=3))
    return client, clusters, jobs


class TestDataprocClient:
    def test_create_cluster_uses_config(
        self, dataproc: tuple[DataprocClient, MagicMock, MagicMock]
    ) -> None:
        client, clusters, _ = dataproc
        client.create_cluster()
        spec = clusters.create_cluster.call_args.kwargs["cluster"]
        assert spec["config"]["worker_config"]["num_instances"] == 3
        assert spec["config"]["software_config"]["optional_components"] == [
            dataproc_v1.Component.JUPYTER
        ]
        clusters.create_cluster.return_value.result.assert_called_once()

    def test_delete_cluster(
        self, dataproc: tuple[DataprocClient, MagicMock, MagicMock]
    ) -> None:
        client, clusters, _ = dataproc
        client.delete_cluster()
        clusters.delete_cluster.return_value.result.assert_called_once()

    def test_cluster_exists(
        self, dataproc: tuple[DataprocClient, MagicMock, MagicMock]
    ) -> None:
        client, clusters, _ = dataproc
        assert client.cluster_exists() is True
        clusters.get_cluster.side_effect = _not_found()
        assert client.cluster_exists() is False

    def test_submit_waits_and_returns_job_id(
        self, dataproc: tuple[DataprocClient, MagicMock, MagicMock]
    ) -> None:
        client, _, jobs = dataproc
        operation = jobs.submit_job_as_operation.return_value
        operation.result.return_value.reference.job_id = "job-1"

        job_id = client.submit_pyspark_job(
            "gs://b/main.py",
            args=["--x"],
            python_file_uris=["gs://b/lib.py"],
            jar_file_uris=["gs://b/extra.jar"],
        )

        assert job_id == "job-1"
        job = jobs.submit_job_as_operation.call_args.kwargs["job"]["pyspark_job"]
        assert job["jar_file_uris"] == [SPARK_BIGQUERY_JAR, "gs://b/extra.jar"]
        assert job["args"] == ["--x"]
        assert job["python_file_uris"] == ["gs://b/lib.py"]

    def test_submit_without_waiting_reads_metadata(
        self, dataproc: tuple[DataprocClient, MagicMock, MagicMock]
    ) -> None:
        client, _, jobs = dataproc
        jobs.submit_job_as_operation.return_value.metadata.job_id = "job-2"
        assert client.submit_pyspark_job("gs://b/main.py", wait=False) == "job-2"
        jobs.submit_job_as_operation.return_value.result.assert_not_called()

    def test_wait_for_job_polls_until_terminal(
        self,
        dataproc: tuple[DataprocClient, MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, _, jobs = dataproc
        states = iter(
            [
                dataproc_v1.JobStatus.State.PENDING,
                dataproc_v1.JobStatus.State.RUNNING,
                dataproc_v1.JobStatus.State.DONE,
            ]
        )

        def get_job(**_: object) -> MagicMock:
            job = MagicMock()
            job.status.state = next(states)
            return job

        jobs.get_job.side_effect = get_job
        sleeps: list[float] = []
        monkeypatch.setattr("tenacity.nap.time.sleep", sleeps.append)

        assert client.wait_for_job("job-1", poll_interval=5) == "DONE"
        assert sleeps == [5, 5]
