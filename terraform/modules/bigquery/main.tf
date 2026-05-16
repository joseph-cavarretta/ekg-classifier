resource "google_bigquery_dataset" "electrocardiograms" {
  dataset_id = var.dataset_id
  project    = var.project_id
  location   = var.location

  description = "EKG classification training and test data"

  labels = var.labels

  delete_contents_on_destroy = false
}
