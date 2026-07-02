# ── Main data bucket ──
resource "google_storage_bucket" "data" {
  name          = var.bucket_name
  location      = var.region
  storage_class = "STANDARD"
  force_destroy = false

  uniform_bucket_level_access = true
}

# ── Service account access (CFs need read/write on GCS) ──
resource "google_storage_bucket_iam_member" "sa_object_admin" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${local.service_account_email}"
}

# ── User access (required for Terraform + manual ops) ──
resource "google_storage_bucket_iam_member" "user_admin" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "user:timothee.cloupmartin@gmail.com"
}

# ── Terraform state bucket (create manually once) ──
# gsutil mb -l us-east1 gs://cvee-20260208-tfstate
# Note: cannot manage its own backend bucket with Terraform
