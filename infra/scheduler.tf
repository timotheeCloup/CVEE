# ── Cloud Scheduler — Single trigger for the ETL workflow ──

resource "google_cloud_scheduler_job" "etl_workflow" {
  name        = "cvee-etl-workflow-trigger"
  description = "Trigger ETL workflow every day at 21:00 UTC"
  schedule    = "0 21 * * *"
  time_zone   = "UTC"
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/${google_workflows_workflow.etl_pipeline.id}/executions"

    # Google APIs (workflowexecutions.googleapis.com) require an OAuth token,
    # not an OIDC token. Using OIDC here returns 401 UNAUTHENTICATED.
    oauth_token {
      service_account_email = local.service_account_email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }
}
