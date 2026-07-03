# ── Single JSON secret (all CF config) ──
resource "google_secret_manager_secret" "cvee" {
  secret_id = "cvee-secrets"
  replication {
    auto {}
  }
}

locals {
  cvee_config = {
    FT_CLIENT_ID      = var.ft_client_id
    FT_CLIENT_SECRET  = var.ft_client_secret
    GCS_BUCKET_NAME   = var.bucket_name
    GCP_PROJECT_ID    = var.project_id
    SB_HOST           = var.sb_host
    SB_PORT           = var.sb_port
    SB_NAME           = var.sb_name
    SB_USER           = var.sb_user
    SB_PASSWORD       = var.sb_password
    DB_HOST           = var.db_host
    DB_PORT           = var.db_port
    DB_NAME           = var.db_name
    DB_USER           = var.db_user
    DB_PASSWORD       = var.db_password
    EMBEDDING_API_URL = var.embedding_api_url
  }
}

resource "google_secret_manager_secret_version" "cvee_v1" {
  secret      = google_secret_manager_secret.cvee.id
  secret_data = jsonencode(local.cvee_config)

  lifecycle {
    ignore_changes = [secret_data] # managed manually via .env
  }
}

resource "google_secret_manager_secret_iam_member" "cvee_access" {
  secret_id = google_secret_manager_secret.cvee.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${local.service_account_email}"
}
