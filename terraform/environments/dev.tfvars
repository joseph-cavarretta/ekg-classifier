# Development environment configuration
# Usage: terraform plan -var-file=environments/dev.tfvars

project_id  = "your-dev-project-id"
region      = "us-central1"
location    = "US"
bucket_name = "your-dev-bucket-name"
dataset_id  = "electrocardiograms"

cluster_name         = "ekg-classifier-dev"
service_account_name = "ekg-classifier-sa-dev"

master_config = {
  machine_type  = "n1-standard-2"
  boot_disk_gb  = 30
  num_instances = 1
}

worker_config = {
  machine_type  = "n1-standard-2"
  boot_disk_gb  = 30
  num_instances = 2
}

labels = {
  project     = "ekg-classifier"
  managed_by  = "terraform"
  environment = "dev"
}
