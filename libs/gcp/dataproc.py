import logging
import time
from typing import Any

from google.api_core.exceptions import NotFound
from google.cloud import dataproc_v1

from config import DataprocConfig, GCPConfig

logger = logging.getLogger(__name__)

SPARK_BIGQUERY_JAR = (
    "gs://spark-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.35.0.jar"
)


class DataprocClient:
    """Dataproc cluster and job management."""

    def __init__(self, gcp_config: GCPConfig, dataproc_config: DataprocConfig) -> None:
        self.gcp_config = gcp_config
        self.dataproc_config = dataproc_config
        self._cluster_client = dataproc_v1.ClusterControllerClient(
            client_options={
                "api_endpoint": f"{gcp_config.region}-dataproc.googleapis.com:443"
            }
        )
        self._job_client = dataproc_v1.JobControllerClient(
            client_options={
                "api_endpoint": f"{gcp_config.region}-dataproc.googleapis.com:443"
            }
        )

    def create_cluster(self) -> None:
        """Create a Dataproc cluster."""
        optional_components = []
        if self.dataproc_config.enable_jupyter:
            optional_components.append(dataproc_v1.Component.JUPYTER)

        cluster_config = {
            "project_id": self.gcp_config.project_id,
            "cluster_name": self.dataproc_config.cluster_name,
            "config": {
                "master_config": {
                    "num_instances": 1,
                    "machine_type_uri": self.dataproc_config.master_machine_type,
                    "disk_config": {
                        "boot_disk_size_gb": self.dataproc_config.master_boot_disk_gb,
                    },
                },
                "worker_config": {
                    "num_instances": self.dataproc_config.num_workers,
                    "machine_type_uri": self.dataproc_config.worker_machine_type,
                    "disk_config": {
                        "boot_disk_size_gb": self.dataproc_config.worker_boot_disk_gb,
                    },
                },
                "software_config": {
                    "image_version": self.dataproc_config.image_version,
                    "optional_components": optional_components,
                },
                "gce_cluster_config": {
                    "internal_ip_only": False,
                },
            },
        }

        logger.info(f"Creating cluster: {self.dataproc_config.cluster_name}")
        operation = self._cluster_client.create_cluster(
            project_id=self.gcp_config.project_id,
            region=self.gcp_config.region,
            cluster=cluster_config,
        )
        operation.result()
        logger.info(f"Cluster created: {self.dataproc_config.cluster_name}")

    def delete_cluster(self) -> None:
        """Delete the Dataproc cluster."""
        logger.info(f"Deleting cluster: {self.dataproc_config.cluster_name}")
        operation = self._cluster_client.delete_cluster(
            project_id=self.gcp_config.project_id,
            region=self.gcp_config.region,
            cluster_name=self.dataproc_config.cluster_name,
        )
        operation.result()
        logger.info(f"Cluster deleted: {self.dataproc_config.cluster_name}")

    def cluster_exists(self) -> bool:
        """Check if the cluster exists."""
        try:
            self._cluster_client.get_cluster(
                project_id=self.gcp_config.project_id,
                region=self.gcp_config.region,
                cluster_name=self.dataproc_config.cluster_name,
            )
            return True
        except NotFound:
            return False

    def submit_pyspark_job(
        self,
        main_python_file_uri: str,
        *,
        args: list[str] | None = None,
        python_file_uris: list[str] | None = None,
        jar_file_uris: list[str] | None = None,
        wait: bool = True,
    ) -> str:
        """Submit a PySpark job to the cluster.

        Args:
            main_python_file_uri: GCS URI of the main Python file
            args: Arguments to pass to the Python script
            python_file_uris: Additional Python files to include
            jar_file_uris: JAR files to include (spark-bigquery added by default)
            wait: Wait for job completion

        Returns:
            Job ID
        """
        jars = [SPARK_BIGQUERY_JAR]
        if jar_file_uris:
            jars.extend(jar_file_uris)

        job: dict[str, Any] = {
            "placement": {"cluster_name": self.dataproc_config.cluster_name},
            "pyspark_job": {
                "main_python_file_uri": main_python_file_uri,
                "jar_file_uris": jars,
            },
        }

        if args:
            job["pyspark_job"]["args"] = args
        if python_file_uris:
            job["pyspark_job"]["python_file_uris"] = python_file_uris

        logger.info(f"Submitting PySpark job: {main_python_file_uri}")
        operation = self._job_client.submit_job_as_operation(
            project_id=self.gcp_config.project_id,
            region=self.gcp_config.region,
            job=job,
        )

        if wait:
            result = operation.result()
            job_id = result.reference.job_id
            logger.info(f"Job completed: {job_id}")
            return job_id

        # return the job id from the operation metadata
        job_id = operation.metadata.job_id
        logger.info(f"Job submitted: {job_id}")
        return job_id

    def get_job_status(self, job_id: str) -> str:
        """Get the status of a job."""
        job = self._job_client.get_job(
            project_id=self.gcp_config.project_id,
            region=self.gcp_config.region,
            job_id=job_id,
        )
        return dataproc_v1.JobStatus.State(job.status.state).name

    def wait_for_job(self, job_id: str, poll_interval: int = 10) -> str:
        """Wait for a job to complete.

        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between status checks

        Returns:
            Final job state
        """
        while True:
            status = self.get_job_status(job_id)
            logger.info(f"Job {job_id} status: {status}")

            if status in ("DONE", "ERROR", "CANCELLED"):
                return status

            time.sleep(poll_interval)
