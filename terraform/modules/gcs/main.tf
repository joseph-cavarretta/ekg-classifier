resource "google_storage_bucket" "data" {
  name     = var.bucket_name
  project  = var.project_id
  location = var.location

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  labels = var.labels
}

resource "google_storage_bucket_object" "data_folder" {
  name    = "electrocardiograms/data/"
  content = " "
  bucket  = google_storage_bucket.data.name
}

resource "google_storage_bucket_object" "models_folder" {
  name    = "electrocardiograms/models/"
  content = " "
  bucket  = google_storage_bucket.data.name
}
