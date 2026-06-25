# ── Main data bucket ──
resource "google_storage_bucket" "data" {
  name          = var.bucket_name
  location      = var.region
  storage_class = "STANDARD"
  force_destroy = false

  uniform_bucket_level_access = true
}

# ── Terraform state bucket (create manually once) ──
# gsutil mb -l us-east1 gs://cvee-20260208-tfstate
# Note: cannot manage its own backend bucket with Terraform
