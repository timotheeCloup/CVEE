terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }

  # Separate state from the production stack (infra/ prefix "cvee").
  backend "gcs" {
    bucket = "cvee-20260208-tfstate"
    prefix = "cvee/dev"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_project" "project" {}

locals {
  # Cloud Run services run as the default compute service account.
  service_account_email = "${data.google_project.project.number}-compute@developer.gserviceaccount.com"
  image_base            = "europe-west1-docker.pkg.dev/${var.project_id}/cvee"
}
