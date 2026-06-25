variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "cvee-20260208"
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-east1"
}

variable "bucket_name" {
  description = "GCS bucket for data storage and function sources"
  type        = string
  default     = "cvee-20260208"
}

# ── France Travail API ──
variable "ft_client_id" {
  description = "France Travail API client ID"
  type        = string
  sensitive   = true
}

variable "ft_client_secret" {
  description = "France Travail API client secret"
  type        = string
  sensitive   = true
}

# ── Supabase ──
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

# ── Local dev (not used in production CFs) ──
variable "db_host" {
  description = "Local dev DB host"
  type        = string
  default     = "localhost"
}

variable "db_port" {
  description = "Local dev DB port"
  type        = string
  default     = "5433"
}

variable "db_name" {
  description = "Local dev DB name"
  type        = string
  default     = "cvee_db"
}

variable "db_user" {
  description = "Local dev DB user"
  type        = string
  default     = "postgres"
}

variable "db_password" {
  description = "Local dev DB password"
  type        = string
  default     = "postgres"
}

# ── Other ──
variable "embedding_api_url" {
  description = "Internal embedding API URL (local dev)"
  type        = string
  default     = "http://embedding-service:8000/embed"
}
