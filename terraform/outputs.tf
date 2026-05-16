output "bucket_name" {
  description = "Name of the GCS bucket"
  value       = module.gcs.bucket_name
}

output "bucket_url" {
  description = "URL of the GCS bucket"
  value       = module.gcs.bucket_url
}

output "dataset_id" {
  description = "BigQuery dataset ID"
  value       = module.bigquery.dataset_id
}

output "dataset_full_id" {
  description = "Full BigQuery dataset ID (project.dataset)"
  value       = module.bigquery.dataset_full_id
}

output "cluster_name" {
  description = "Dataproc cluster name"
  value       = module.dataproc.cluster_name
}

output "service_account_email" {
  description = "Service account email"
  value       = module.iam.service_account_email
}
