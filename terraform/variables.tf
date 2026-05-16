variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for regional resources"
  type        = string
  default     = "us-central1"
}

variable "location" {
  description = "GCP location for multi-regional resources (GCS, BigQuery)"
  type        = string
  default     = "US"
}

variable "bucket_name" {
  description = "Name of the GCS bucket for data and models"
  type        = string
}

variable "dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
  default     = "electrocardiograms"
}

variable "cluster_name" {
  description = "Dataproc cluster name"
  type        = string
  default     = "ekg-classifier-cluster"
}

variable "service_account_name" {
  description = "Name of the service account for EKG classifier"
  type        = string
  default     = "ekg-classifier-sa"
}

variable "master_config" {
  description = "Dataproc master node configuration"
  type = object({
    machine_type  = string
    boot_disk_gb  = number
    num_instances = number
  })
  default = {
    machine_type  = "n1-standard-2"
    boot_disk_gb  = 30
    num_instances = 1
  }
}

variable "worker_config" {
  description = "Dataproc worker node configuration"
  type = object({
    machine_type  = string
    boot_disk_gb  = number
    num_instances = number
  })
  default = {
    machine_type  = "n1-standard-2"
    boot_disk_gb  = 30
    num_instances = 2
  }
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    project     = "ekg-classifier"
    managed_by  = "terraform"
    environment = "dev"
  }
}
