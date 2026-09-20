# ── Cloud Run: dev API (FastAPI) ──
# Private service (no allUsers binding): only the invoker_user and the UI
# service account can call it. Access from a browser is done through
# `gcloud run services proxy cvee-api-dev`.
resource "google_cloud_run_v2_service" "api_dev" {
  name     = "cvee-api-dev"
  location = var.cloud_run_region
  project  = var.project_id

  template {
    containers {
      image = "${local.image_base}/cvee-api:${var.image_tag}"
      env {
        name  = "DB_HOST"
        value = var.sb_host
      }
      env {
        name  = "DB_PORT"
        value = var.sb_port
      }
      env {
        name  = "DB_NAME"
        value = var.sb_name
      }
      env {
        name  = "DB_USER"
        value = var.sb_user
      }
      env {
        name  = "DB_PASSWORD"
        value = var.sb_password
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "2048Mi"
        }
        startup_cpu_boost = true
      }
    }
    timeout = "300s"
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }
  }

  # The image is owned by CI (gcloud run deploy), not Terraform: only the
  # initial bootstrap uses var.image_tag. Without this, any terraform apply
  # would revert the image to the bootstrap tag.
  lifecycle {
    ignore_changes = [template[0].containers[0].image]
  }
}

# ── Cloud Run: dev UI (Streamlit) ──
resource "google_cloud_run_v2_service" "ui_dev" {
  name     = "cvee-ui-dev"
  location = var.cloud_run_region
  project  = var.project_id

  template {
    containers {
      image = "${local.image_base}/cvee-ui:${var.image_tag}"
      env {
        name  = "API_URL"
        value = "${google_cloud_run_v2_service.api_dev.uri}/embed-cv"
      }
      env {
        name  = "API_AUTH"
        value = "true"
      }
    }
    timeout = "300s"
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }
  }

  lifecycle {
    ignore_changes = [template[0].containers[0].image]
  }
}

# ── IAM: private access ──

# The developer can invoke both services (via gcloud run services proxy).
resource "google_cloud_run_v2_service_iam_member" "api_dev_invoker_user" {
  name     = google_cloud_run_v2_service.api_dev.name
  location = google_cloud_run_v2_service.api_dev.location
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "user:${var.invoker_user}"
}

resource "google_cloud_run_v2_service_iam_member" "ui_dev_invoker_user" {
  name     = google_cloud_run_v2_service.ui_dev.name
  location = google_cloud_run_v2_service.ui_dev.location
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "user:${var.invoker_user}"
}

# The UI service account can call the API (service-to-service identity token).
resource "google_cloud_run_v2_service_iam_member" "api_dev_invoker_ui_sa" {
  name     = google_cloud_run_v2_service.api_dev.name
  location = google_cloud_run_v2_service.api_dev.location
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "serviceAccount:${local.service_account_email}"
}
