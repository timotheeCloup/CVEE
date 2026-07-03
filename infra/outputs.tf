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

output "cvee_api_url" {
  description = "CVEE FastAPI service URL"
  value       = google_cloud_run_v2_service.api.uri
}

output "cvee_ui_url" {
  description = "CVEE Streamlit UI URL"
  value       = google_cloud_run_v2_service.ui.uri
}

output "bucket_name" {
  description = "GCS data bucket"
  value       = google_storage_bucket.data.name
}

output "secret_name" {
  description = "Secret Manager secret ID"
  value       = google_secret_manager_secret.cvee.secret_id
}

output "workflow_name" {
  description = "Cloud Workflow ETL pipeline name"
  value       = google_workflows_workflow.etl_pipeline.name
}
