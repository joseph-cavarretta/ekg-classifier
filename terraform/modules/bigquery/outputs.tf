output "dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.electrocardiograms.dataset_id
}

output "dataset_full_id" {
  description = "Full dataset ID (project.dataset)"
  value       = "${var.project_id}.${google_bigquery_dataset.electrocardiograms.dataset_id}"
}

output "dataset_self_link" {
  description = "Self link of the dataset"
  value       = google_bigquery_dataset.electrocardiograms.self_link
}
