# ── Cloud Workflows — Serverless ETL orchestrator ──
# Replaces 3 independent Cloud Scheduler jobs with a single Workflow
# Free tier: 5,000 steps/month (we use ~270/month)

resource "google_workflows_workflow" "etl_pipeline" {
  name        = "cvee-etl-pipeline"
  region      = var.region
  description = "Orchestrate api-to-gcs → pipeline → ingest-db daily ETL"

  service_account = local.service_account_email

  source_contents = templatefile("${path.module}/workflows/etl_pipeline.yaml.tftpl", {
    api_to_gcs_url = google_cloudfunctions2_function.api_to_gcs.service_config[0].uri
    pipeline_url   = google_cloudfunctions2_function.pipeline.service_config[0].uri
    ingest_db_url  = google_cloudfunctions2_function.ingest_db.service_config[0].uri
  })

  depends_on = [
    google_cloudfunctions2_function.api_to_gcs,
    google_cloudfunctions2_function.pipeline,
    google_cloudfunctions2_function.ingest_db,
  ]
}

# Allow Cloud Scheduler (compute SA) to trigger Cloud Workflows
resource "google_project_iam_member" "workflows_invoker" {
  project = var.project_id
  role    = "roles/workflows.invoker"
  member  = "serviceAccount:${local.service_account_email}"
}
