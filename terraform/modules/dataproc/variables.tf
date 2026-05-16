variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "cluster_name" {
  description = "Dataproc cluster name"
  type        = string
}

variable "master_config" {
  description = "Master node configuration"
  type = object({
    machine_type  = string
    boot_disk_gb  = number
    num_instances = number
  })
}

variable "worker_config" {
  description = "Worker node configuration"
  type = object({
    machine_type  = string
    boot_disk_gb  = number
    num_instances = number
  })
}

variable "service_account_email" {
  description = "Service account email for the cluster"
  type        = string
}

variable "labels" {
  description = "Labels to apply to the cluster"
  type        = map(string)
  default     = {}
}
