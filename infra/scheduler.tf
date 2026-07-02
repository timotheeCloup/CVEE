# ── Cloud Scheduler jobs ──

resource "google_cloud_scheduler_job" "api_to_gcs" {
  name        = "api-to-gcs-scheduler"
  description = "Trigger api-to-gcs-cf every day at 21:00 UTC"
  schedule    = "0 21 * * *"
  time_zone   = "UTC"
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions2_function.api_to_gcs.service_config[0].uri

    oidc_token {
      service_account_email = local.service_account_email
      audience              = google_cloudfunctions2_function.api_to_gcs.service_config[0].uri
    }
  }
}

resource "google_cloud_scheduler_job" "pipeline" {
  name        = "pipeline-scheduler"
  description = "Trigger pipeline-cf every day at 21:30 UTC"
  schedule    = "30 21 * * *"
  time_zone   = "UTC"
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions2_function.pipeline.service_config[0].uri

    oidc_token {
      service_account_email = local.service_account_email
      audience              = google_cloudfunctions2_function.pipeline.service_config[0].uri
    }
  }
}

resource "google_cloud_scheduler_job" "ingest_db" {
  name        = "ingest-db-scheduler"
  description = "Trigger ingest-db-cf every day at 22:00 UTC"
  schedule    = "0 22 * * *"
  time_zone   = "UTC"
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions2_function.ingest_db.service_config[0].uri

    oidc_token {
      service_account_email = local.service_account_email
      audience              = google_cloudfunctions2_function.ingest_db.service_config[0].uri
    }
  }
}
