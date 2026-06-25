output "api_to_gcs_url" {
  description = "api-to-gcs-cf trigger URL"
  value       = google_cloudfunctions2_function.api_to_gcs.service_config[0].uri
}

output "pipeline_url" {
  description = "pipeline-cf trigger URL"
  value       = google_cloudfunctions2_function.pipeline.service_config[0].uri
}

output "ingest_db_url" {
  description = "ingest-db-cf trigger URL"
  value       = google_cloudfunctions2_function.ingest_db.service_config[0].uri
}

output "bucket_name" {
  description = "GCS data bucket"
  value       = google_storage_bucket.data.name
}

output "secret_name" {
  description = "Secret Manager secret ID"
  value       = google_secret_manager_secret.cvee.secret_id
}
