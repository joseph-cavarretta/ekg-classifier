resource "google_dataproc_cluster" "ekg_classifier" {
  name    = var.cluster_name
  project = var.project_id
  region  = var.region

  labels = var.labels

  cluster_config {
    master_config {
      num_instances = var.master_config.num_instances
      machine_type  = var.master_config.machine_type

      disk_config {
        boot_disk_size_gb = var.master_config.boot_disk_gb
        boot_disk_type    = "pd-standard"
      }
    }

    worker_config {
      num_instances = var.worker_config.num_instances
      machine_type  = var.worker_config.machine_type

      disk_config {
        boot_disk_size_gb = var.worker_config.boot_disk_gb
        boot_disk_type    = "pd-standard"
      }
    }

    software_config {
      image_version = "2.1-debian11"

      optional_components = ["JUPYTER"]

      override_properties = {
        "spark:spark.jars.packages" = "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.35.0"
      }
    }

    gce_cluster_config {
      service_account = var.service_account_email

      service_account_scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
      ]
    }

    endpoint_config {
      enable_http_port_access = true
    }
  }
}
