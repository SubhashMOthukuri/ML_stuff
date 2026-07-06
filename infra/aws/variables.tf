variable "region" {
  type    = string
  default = "us-east-2"
}

variable "app_name" {
  type    = string
  default = "nyc-airbnb"
}

variable "redis_password" {
  type      = string
  sensitive = true
}

variable "valid_api_keys" {
  type      = string
  sensitive = true
}

variable "slack_webhook_url" {
  type      = string
  sensitive = true
  default   = ""
}

variable "dd_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

variable "ground_truth_ingest_token" {
  type      = string
  sensitive = true
  default   = ""
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "Master password for RDS PostgreSQL instance"
}
