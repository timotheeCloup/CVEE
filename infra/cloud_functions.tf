# ── Zip function sources ──
data "archive_file" "api_to_gcs_source" {
  type        = "zip"
  source_dir  = "${path.module}/../functions/api-to-gcs"
  output_path = "${path.module}/.tmp/api-to-gcs.zip"
}

data "archive_file" "pipeline_source" {
  type        = "zip"
  source_dir  = "${path.module}/../functions/pipeline"
  output_path = "${path.module}/.tmp/pipeline.zip"
}

data "archive_file" "ingest_db_source" {
  type        = "zip"
  source_dir  = "${path.module}/../functions/ingest-db"
  output_path = "${path.module}/.tmp/ingest-db.zip"
}

# ── Upload zips to GCS ──
resource "google_storage_bucket_object" "api_to_gcs_zip" {
  name   = "sources/api-to-gcs-${data.archive_file.api_to_gcs_source.output_sha256}.zip"
  bucket = google_storage_bucket.data.name
  source = data.archive_file.api_to_gcs_source.output_path
}

resource "google_storage_bucket_object" "pipeline_zip" {
  name   = "sources/pipeline-${data.archive_file.pipeline_source.output_sha256}.zip"
  bucket = google_storage_bucket.data.name
  source = data.archive_file.pipeline_source.output_path
}

resource "google_storage_bucket_object" "ingest_db_zip" {
  name   = "sources/ingest-db-${data.archive_file.ingest_db_source.output_sha256}.zip"
  bucket = google_storage_bucket.data.name
  source = data.archive_file.ingest_db_source.output_path
}

# ── Cloud Functions (gen2) ──

resource "google_cloudfunctions2_function" "api_to_gcs" {
  name        = "api-to-gcs-cf"
  location    = var.region
  description = "Fetch France Travail jobs → GCS (daily 21:00)"

  build_config {
    runtime     = "python312"
    entry_point = "api_to_gcs_cf"
    source {
      storage_source {
        bucket = google_storage_bucket.data.name
        object = google_storage_bucket_object.api_to_gcs_zip.name
      }
    }
  }

  service_config {
    max_instance_count = 1
    available_memory   = "512M"
    timeout_seconds    = 540
  }

  depends_on = [google_secret_manager_secret_version.cvee_v1]
}

resource "google_cloudfunctions2_function" "pipeline" {
  name        = "pipeline-cf"
  location    = var.region
  description = "Bronze→Silver→Gold ETL (daily 21:30)"

  build_config {
    runtime     = "python312"
    entry_point = "pipeline_cf"
    source {
      storage_source {
        bucket = google_storage_bucket.data.name
        object = google_storage_bucket_object.pipeline_zip.name
      }
    }
  }

  service_config {
    max_instance_count = 1
    available_memory   = "2048M"
    timeout_seconds    = 1800
  }

  depends_on = [google_secret_manager_secret_version.cvee_v1]
}

resource "google_cloudfunctions2_function" "ingest_db" {
  name        = "ingest-db-cf"
  location    = var.region
  description = "GCS → Supabase + cleanup (daily 23:30)"

  build_config {
    runtime     = "python312"
    entry_point = "ingest_db_cf"
    source {
      storage_source {
        bucket = google_storage_bucket.data.name
        object = google_storage_bucket_object.ingest_db_zip.name
      }
    }
  }

  service_config {
    max_instance_count = 1
    available_memory   = "1024M"
    timeout_seconds    = 3600
  }

  depends_on = [google_secret_manager_secret_version.cvee_v1]
}
