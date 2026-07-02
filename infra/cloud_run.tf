# ── Artifact Registry ──
resource "google_artifact_registry_repository" "cvee" {
  location      = "europe-west1"
  repository_id = "cvee"
  format        = "DOCKER"
  project       = var.project_id
}

# ── Cloud Run: API (FastAPI) ──
resource "google_cloud_run_v2_service" "api" {
  name     = "cvee-api"
  location = var.cloud_run_region
  project  = var.project_id

  template {
    containers {
      image = "europe-west1-docker.pkg.dev/${var.project_id}/cvee/cvee-api:latest"
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
      }
    }
    timeout = "300s"
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }
  }
}

# ── Cloud Run: UI (Streamlit) ──
resource "google_cloud_run_v2_service" "ui" {
  name     = "cvee-ui"
  location = var.cloud_run_region
  project  = var.project_id

  template {
    containers {
      image = "europe-west1-docker.pkg.dev/${var.project_id}/cvee/cvee-ui:latest"
      env {
        name  = "API_URL"
        value = "${google_cloud_run_v2_service.api.uri}/embed-cv"
      }
    }
    timeout = "300s"
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }
  }
}

# ── IAM: allow unauthenticated HTTP triggers ──
resource "google_cloud_run_v2_service_iam_member" "api_invoker" {
  name     = google_cloud_run_v2_service.api.name
  location = google_cloud_run_v2_service.api.location
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "ui_invoker" {
  name     = google_cloud_run_v2_service.ui.name
  location = google_cloud_run_v2_service.ui.location
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "allUsers"
}
