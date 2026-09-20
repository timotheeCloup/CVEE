variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "cvee-20260208"
}

variable "region" {
  description = "GCP region for the provider (not used by Cloud Run here)"
  type        = string
  default     = "us-east1"
}

variable "cloud_run_region" {
  description = "GCP region for the dev Cloud Run services"
  type        = string
  default     = "europe-west1"
}

# ── Supabase (same database as production, read-only from the API) ──
variable "sb_host" {
  description = "Supabase host"
  type        = string
  default     = "aws-1-eu-west-3.pooler.supabase.com"
}

variable "sb_port" {
  description = "Supabase port"
  type        = string
  default     = "5432"
}

variable "sb_name" {
  description = "Supabase database name"
  type        = string
  default     = "postgres"
}

variable "sb_user" {
  description = "Supabase user"
  type        = string
}

variable "sb_password" {
  description = "Supabase password"
  type        = string
  sensitive   = true
}

# ── Access control ──
variable "invoker_user" {
  description = "Google account allowed to invoke the private dev services"
  type        = string
}

variable "image_tag" {
  description = "Image tag used for the initial service bootstrap (overridden by CI on each deploy)"
  type        = string
  default     = "latest"
}
