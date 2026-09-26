# ── Cloud Monitoring — alerting on failed ETL workflow executions ──
# Free of charge: alerting policies and email notification channels are not billed.
# We alert on the daily ETL workflow finishing in a FAILED state, so a silent
# pipeline failure (like the 2026-09 OOM streak) can no longer go unnoticed.

# Email channel (recipient set via var.alert_email).
resource "google_monitoring_notification_channel" "email_alert" {
  display_name = "CVEE alerts (email)"
  project      = var.project_id
  type         = "email"

  labels = {
    email_address = var.alert_email
  }
}

# One alert policy per condition: any FAILED execution of the ETL workflow.
resource "google_monitoring_alert_policy" "etl_workflow_failed" {
  display_name = "CVEE ETL workflow failed"
  project      = var.project_id
  combiner     = "OR"
  severity     = "ERROR"

  documentation {
    mime_type = "text/markdown"
    content   = <<-EOT
      The daily CVEE ETL workflow (`cvee-etl-pipeline`) finished in a FAILED state.

      Check the failing step and its logs:

      ```bash
      gcloud workflows executions list cvee-etl-pipeline \
        --location=${var.region} --project=${var.project_id} --limit=3

      gcloud logging read 'resource.labels.service_name="pipeline-cf"' \
        --project=${var.project_id} --limit=50 --freshness=1d
      ```

      Common causes: `pipeline-cf` OOM (see functions/pipeline/core.py batch size),
      ingest timeout, or an upstream France Travail API failure.
    EOT
  }

  conditions {
    display_name = "ETL workflow finished in FAILED state"

    condition_threshold {
      # `finished_execution_count` is a DELTA counter with a `status` label.
      # Summing over the alignment window yields the number of failed runs;
      # anything above zero triggers the alert.
      filter          = "resource.type = \"workflows.googleapis.com/Workflow\" AND resource.labels.workflow_id = \"${google_workflows_workflow.etl_pipeline.name}\" AND metric.type = \"workflows.googleapis.com/finished_execution_count\" AND metric.labels.status = \"FAILED\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"

      trigger {
        count = 1
      }

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email_alert.id]

  # Auto-close the incident once a subsequent run succeeds.
  alert_strategy {
    auto_close = "1800s"
  }
}
