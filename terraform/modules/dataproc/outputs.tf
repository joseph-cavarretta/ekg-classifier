output "cluster_name" {
  description = "Name of the Dataproc cluster"
  value       = google_dataproc_cluster.ekg_classifier.name
}

output "cluster_master_instance_name" {
  description = "Name of the master instance"
  value       = google_dataproc_cluster.ekg_classifier.cluster_config[0].master_config[0].instance_names[0]
}
