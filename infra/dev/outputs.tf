output "api_dev_url" {
  description = "Dev API service URL (private)"
  value       = google_cloud_run_v2_service.api_dev.uri
}

output "ui_dev_url" {
  description = "Dev UI service URL (private)"
  value       = google_cloud_run_v2_service.ui_dev.uri
}

output "ui_dev_proxy_command" {
  description = "Command to open the private dev UI locally in a browser"
  value       = "gcloud run services proxy cvee-ui-dev --project ${var.project_id} --region ${var.cloud_run_region} --port 8080"
}
