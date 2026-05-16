output "service_account_email" {
  description = "Email of the service account"
  value       = google_service_account.ekg_classifier.email
}

output "service_account_name" {
  description = "Name of the service account"
  value       = google_service_account.ekg_classifier.name
}
