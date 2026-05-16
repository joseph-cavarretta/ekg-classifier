terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

module "gcs" {
  source = "./modules/gcs"

  project_id  = var.project_id
  bucket_name = var.bucket_name
  location    = var.location
  labels      = var.labels
}

module "bigquery" {
  source = "./modules/bigquery"

  project_id = var.project_id
  dataset_id = var.dataset_id
  location   = var.location
  labels     = var.labels
}

module "iam" {
  source = "./modules/iam"

  project_id           = var.project_id
  service_account_name = var.service_account_name
  bucket_name          = module.gcs.bucket_name
  dataset_id           = module.bigquery.dataset_id
}

module "dataproc" {
  source = "./modules/dataproc"

  project_id    = var.project_id
  region        = var.region
  cluster_name  = var.cluster_name
  master_config = var.master_config
  worker_config = var.worker_config
  labels        = var.labels

  service_account_email = module.iam.service_account_email

  depends_on = [module.iam]
}
