terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }

  backend "gcs" {
    bucket = "cvee-20260208-tfstate"
    prefix = "cvee"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_project" "project" {}

locals {
  service_account_email = "${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}
