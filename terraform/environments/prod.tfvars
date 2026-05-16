# Production environment configuration
# Usage: terraform plan -var-file=environments/prod.tfvars

project_id  = "your-prod-project-id"
region      = "us-central1"
location    = "US"
bucket_name = "your-prod-bucket-name"
dataset_id  = "electrocardiograms"

cluster_name         = "ekg-classifier-prod"
service_account_name = "ekg-classifier-sa-prod"

master_config = {
  machine_type  = "n1-standard-4"
  boot_disk_gb  = 50
  num_instances = 1
}

worker_config = {
  machine_type  = "n1-standard-4"
  boot_disk_gb  = 50
  num_instances = 4
}

labels = {
  project     = "ekg-classifier"
  managed_by  = "terraform"
  environment = "prod"
}
