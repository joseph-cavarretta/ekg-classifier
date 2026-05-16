variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "service_account_name" {
  description = "Name of the service account"
  type        = string
}

variable "bucket_name" {
  description = "Name of the GCS bucket (for IAM bindings)"
  type        = string
}

variable "dataset_id" {
  description = "BigQuery dataset ID (for IAM bindings)"
  type        = string
}
